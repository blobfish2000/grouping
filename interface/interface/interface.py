import json
import os
import shutil
import socket
import sqlite3
import uuid
from collections import OrderedDict
from datetime import datetime

import click
import numpy as np
import requests
from colorama import init, Fore, Style
from flask import Flask, render_template, g, request, Response, url_for, redirect, make_response

app = Flask(__name__)
app.config.from_object(__name__)

app.config.update(dict(
    DATABASE=os.path.join(app.root_path, 'db', 'interface.db'),
    SECRET_KEY='C796D37C6D491E8F0C6E9B83EED34C15C0F377F9F0F3CBB3216FBBF776DA6325',
    USERNAME='admin',
    PASSWORD='password'
))

DEFAULT_SAMPLES_PER_PARTICIPANT = 500
SAMPLES_URL_PREFIX = 'http://147.182.213.96:8000/'
SAMPLES_ROOT = ''
APP_ROOT = os.path.dirname(os.path.abspath(__file__))  # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')
NO_LABELER_ID = "NO_LABELER_ID"
NOT_MTURK = "NO_MTURK"
EXPERIMENT_ID_NOT_AVAILABLE = "EXPERIMENT_ID_NOT_AVAILABLE"
LABELER_ID_COOKIE_KEY = 'labeler_id'


@app.cli.command('dump_samples')
@click.option('--database', help='database file to use, a *.db file', default=None)
def dumpdb_samples_command(database):
    """ Print the samples table """
    dump_db(None, database, responses=False, samples=True)

@app.cli.command('dumpdb')
@click.option('--outfile', help="name for output file containing the responses database", type=click.Path())
@click.option('--database', help='database file to use, a *.db file', default=None)
def dumpdb_command(outfile, database):
    """ Print the database (can save to CSV) """
    dump_db(outfile, database, responses=True, samples=False)


@app.cli.command('remove_experiment')
@click.argument('experiment_id')
@click.option('--database', help='database file to use, a *.db file', default=None)
def remove_command(experiment_id, database):
    remove_experiment(experiment_id, database)


@app.cli.command('remove_sample')
@click.argument('sample_name')
@click.option('--force/--no-force', help='force remove something even if it has responses', default=False)
def remove_command(sample_name, force):
    remove_from_sample_db(sample_name, force)


@app.cli.command('load')
@click.argument('directory')
@click.option('--database', help='database file to use, a *.db file', default=None)
def load_command(directory, database):
    """ insert ALL the files from the given folder """
    success = load(directory)

    if success:
        dump_db(False, database, responses=False, samples=True)


@app.cli.command('initdb')
@click.option('--database', help='database file to use, a *.db file', default=None)
@click.option('--force/--no-force', help='force initdb, even if on mprlab server', default=False)
def initdb_command(database, force):
    """Initializes the database."""
    success = init_db(database, force)

    if success:
        dump_db(False, database, responses=True, samples=True)


def connect_db(alternate_db_path=None):
    """Connects to the specific database."""
    if alternate_db_path is None:
        print("Using default database path:", app.config['DATABASE'])
        rv = sqlite3.connect(app.config['DATABASE'], detect_types=sqlite3.PARSE_DECLTYPES)
    else:
        rv = sqlite3.connect(alternate_db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    rv.row_factory = sqlite3.Row

    return rv


def get_db(alternate_db_path=None):
    """Opens a new database connection if there is none yet for the
    current application context.
    """
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = connect_db(alternate_db_path)
    return g.sqlite_db


def init_db(alternate_db_path, force):
    db = get_db(alternate_db_path)

    if not force:
        y = input("Are you sure you want to DELETE ALL DATA and re-initialize the database? [y/n]")
        if y != 'y' and y != 'Y':
            print("Aborting.")
            return False

    hostname = socket.gethostname()
    if hostname == "mprlab" and not force:
        print(Fore.RED, "hostname is mprlab. You definitely don't want to do this. Aborting.", Fore.RESET)
        return False

    # apply the schema
    with app.open_resource('schema.sql', mode='r') as f:
        db.cursor().executescript(f.read())

    db.commit()

    print(Fore.RED + "Database Initialized" + Style.RESET_ALL)
    return True


def load(directory):
    db = get_db()

    if not os.path.isdir(directory):
        print(Fore.RED, end='')
        print(directory, " is not a directory. Aborting.")
        print(Fore.RESET, end='')
        return

    abs_path = os.path.abspath(directory)

    if not abs_path.startswith(SAMPLES_ROOT):
        print(Fore.RED, "samples must live in subdirectories in " + SAMPLES_ROOT + ". Aborting.", Fore.RESET)
        return

    subdir = os.path.split(abs_path)[-1]

    # insert ALL the files from the file system
    sample_names = os.listdir(directory)
    for sample_name in sample_names:
        if not sample_name:
            print(Fore.YELLOW, "Skipping empty", Fore.RESET)
            continue
        elif not os.path.isfile(os.path.join(directory, sample_name)):
            print(Fore.YELLOW, "Skipping directory:", Fore.RESET, sample_name)
            continue
        elif not sample_name.endswith("mp3"):
            print(Fore.YELLOW, "Skipping non-mp3:", Fore.RESET, sample_name)
            continue
        else:
            sample_url = os.path.join(SAMPLES_URL_PREFIX, subdir, sample_name)
            try:
                db.execute('INSERT INTO samples (url, count) VALUES (?, 0)', [sample_url])
                print(Fore.BLUE, end='')
                print("Added", sample_url)
                print(Fore.RESET, end='')
            except sqlite3.IntegrityError:
                # skip this because the sample already exists!
                print(Fore.YELLOW, end='')
                print("Skipped", sample_url)
                print(Fore.RESET, end='')

    db.commit()

    return True


def dump_db(outfile_name, database, responses=True, samples=True):
    # for pretty terminal output
    init()

    db = get_db(database)

    if outfile_name:
        outfile = open(outfile_name, 'w')
    else:
        outfile = open(os.devnull, 'w')

    def print_samples_db():
        samples_cur = db.execute('SELECT url FROM samples')
        entries = samples_cur.fetchall()

        # figure out dimensions
        url_w = "100"

        header_format = "{:<" + url_w + "." + url_w + "s}"
        header = header_format.format("URL")
        w = len(header)
        row_format = "{:<" + url_w + "." + url_w + "s}"

        print(Fore.GREEN + "Dumping Database" + Style.RESET_ALL)

        print("=" * w)
        print(header)
        print("=" * w)
        for entry in entries:
            url = entry[0]
            url = url[len(SAMPLES_URL_PREFIX):]
            print(row_format.format(url))
        print("=" * w)

    def print_response_db():
        responses_cur = db.execute(
            'SELECT id, url, stamp, labeler_id, experiment_id, metadata, data FROM responses ORDER BY stamp DESC')
        entries = responses_cur.fetchall()

        json_responses = []

        headers = OrderedDict()
        headers['id'] = 4
        headers['url'] = 25
        headers['stamp'] = 19
        headers['labeler_id'] = 36
        term_size = shutil.get_terminal_size((120, 20))
        total_width = term_size.columns
        headers['data'] = max(total_width - sum(headers.values()) - len(headers), 0)
        fmt = ""
        for k, w in headers.items():
            fmt += "{:<" + str(w) + "." + str(w) + "s} "
        fmt = fmt.strip(' ')
        header = fmt.format(*headers.keys())
        print("=" * total_width)
        print(header)
        for entry in entries:
            response = json.loads(entry[6])
            json_responses.append({
                'id': entry[0],
                'url': entry[1],
                'stamp': str(entry[2]),
                'labeler_id': entry[3],
                'experiment_id': entry[4],
                'metadata': json.loads(entry[5]),
                'data': response,
            })
            cols = [str(col) for col in entry]
            cols[1] = cols[1].strip(SAMPLES_URL_PREFIX)
            cols.pop(4)
            cols.pop(4)
            data = "["
            for d in response['final_response']:
                s = "%0.2f, " % d['timestamp']
                if len(data + s) > headers['data'] - 4:
                    data += "..., "
                    break
                data += s
            if len(data) == 0:
                cols[4] = data + "]"
            else:
                cols[4] = data[:-2] + "]"
            if len(cols[4]) > headers['data']:
                cols[4] = cols[4][0:headers['data'] - 3] + '...'
            print(fmt.format(*cols))
        print("=" * total_width)

        json_out = {'dataset': json_responses}
        json.dump(json_out, outfile, indent=2)

    if samples:
        print_samples_db()
    if responses:
        print_response_db()


def remove_experiment(experiment_id, database):
    db = get_db(database)
    try:
        check_cur = db.execute('SELECT id, stamp, metadata FROM responses WHERE experiment_id=?', [experiment_id])
        check_responses = check_cur.fetchall()
        if check_responses is None or len(check_responses) == 0:
            print(Fore.YELLOW, end='')
            print("experiment with id ", experiment_id, "does not exist.")
            print(Fore.YELLOW, end='')
            return
        else:
            for response in check_responses:
                id = response[0]
                stamp = response[1]
                metadata = response[2]
                assignment_id = json.loads(metadata).get('assignment_id', 'ASSIGNMENT_ID_MISSING')

                k = input("confirm stamp should be " + str(stamp) + "? ")
                if k != 'Y' and k != 'y':
                    print("skipping...")
                    continue
                k = input("confirm assignment ID should be {:s}? ".format(assignment_id))
                if k != 'Y' and k != 'y':
                    print("skipping...")
                    continue

                remove_cur = db.execute('DELETE FROM responses WHERE id=?', [id])

                if remove_cur.rowcount == 1:
                    print(Fore.BLUE, end='')
                    print("Removed {:d} experiments with id {:s}".format(remove_cur.rowcount, experiment_id))
                    print(Fore.RESET, end='')
                else:
                    print(Fore.YELLOW, end='')
                    print("Failed to remove", experiment_id, ". Try again, this might be a race condition.")
                    print(Fore.RESET, end='')

        db.commit()
    except sqlite3.IntegrityError as e:
        print(Fore.RED, end='')
        print(e)
        print(Fore.RESET, end='')


def remove_from_sample_db(sample, force=False):
    db = get_db()
    sample_url = SAMPLES_URL_PREFIX + sample
    try:
        if not force:
            y = input("Are you sure you want to DELETE ALL DATA and re-initialize the database? [y/n]")
            if y != 'y' and y != 'Y':
                remove_cur = db.execute('DELETE FROM samples WHERE url=?', [sample_url])
            else:
                print("Aborting.")
                return
        else:
            remove_cur = db.execute('DELETE FROM samples WHERE url=?', [sample_url])

        if remove_cur.rowcount == 1:
            print(Fore.BLUE, end='')
            print("Removed", sample_url)
            print(Fore.RESET, end='')
        else:
            print(Fore.YELLOW, end='')
            print("Failed to remove", sample_url, ". Try again, this might be a race condition.")
            print(Fore.RESET, end='')
    except sqlite3.IntegrityError as e:
        print(Fore.RED, end='')
        print(e)
        print(Fore.RESET, end='')

    db.commit()


def sample_new_urls(entries, samples_per_participant):
    """ samples from a*x^a-1 """
    a = 10
    sample_indeces = []
    while True:
        idx = int(np.random.power(a) * entries.shape[0])
        if idx not in sample_indeces:
            sample_indeces.append(idx)
        if len(sample_indeces) == samples_per_participant:
            break
    sample_indeces = np.array(sample_indeces)
    return entries[sample_indeces]


@app.teardown_appcontext
def close_db(error):
    """Closes the database again at the end of the request."""
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()


@app.route('/responses', methods=['POST'])
def responses():
    db = get_db()
    labeler_id = request.cookies.get(LABELER_ID_COOKIE_KEY, NO_LABELER_ID)

    if labeler_id == NO_LABELER_ID:
        return Response("No labeler id cookie", status=400, mimetype='application/json')

    req_data = request.get_json()
    sample = req_data['sample']
    ip_addr = request.remote_addr
    stamp = datetime.now()
    metadata = req_data['metadata']
    experiment_id = req_data['experiment_id']

    sample_response = req_data['response']
    url = sample['url']
    # sort the final response by timestamps for sanity
    sorted_final_response = sorted(sample_response['final_response'], key=lambda d: d['timestamp'])
    sample_response['final_response'] = sorted_final_response

    # Insert the full response details
    db.execute(
        'INSERT INTO responses'
        '(url, ip_addr, stamp, labeler_id, experiment_id, metadata, data)'
        'VALUES (?, ?, ?, ?, ?, ?, ?)',
        [url, ip_addr, stamp, labeler_id, experiment_id, json.dumps(metadata), json.dumps(sample_response)])

    # add the labeler if not already present
    try:
        db.execute('INSERT INTO labelers (labeler_id) VALUES (?) ', [labeler_id])
    except sqlite3.IntegrityError:
        # skip this because the sample already exists!
        pass

    db.commit()

    # submit the answers to mechanical turk if necessary as well
    # url = "https://www.mturk.com/mturk/externalSubmit"

    data = {'status': 'ok'}
    js = json.dumps(data)
    resp = Response(js, status=200, mimetype='application/json')
    return resp


@app.route('/id', methods=['GET'])
def id():
    # unique ID for this labeler
    if LABELER_ID_COOKIE_KEY in request.cookies:
        labeler_id = request.cookies[LABELER_ID_COOKIE_KEY]
    else:
        labeler_id = str(uuid.uuid4())

    template = render_template('id.html', labeler_id=labeler_id)
    resp = make_response(template)

    if app.debug:
        max_age_seconds = None  # when debugging, delete cookie when browser exits
    else:
        max_age_seconds = 365 * 24 * 60 * 60  # 1 year

    resp.set_cookie(LABELER_ID_COOKIE_KEY, labeler_id, max_age=max_age_seconds)
    return resp


@app.route('/survey', methods=['GET'])
def survey():
    return redirect('https://goo.gl/forms/CenUVxFWTOlDKrn23')


@app.route('/welcome', methods=['GET'])
def welcome():
    return render_template('welcome.html')


@app.route('/thankyou_mturk', methods=['GET'])
def thank_you_mturk():
    assignment_id = request.args.get('assignmentId', NOT_MTURK)
    experiment_id = request.args.get('experimentId', EXPERIMENT_ID_NOT_AVAILABLE)
    return render_template('thankyou_mturk.html', assignment_id=assignment_id, experiment_id=experiment_id)


@app.route('/thankyou', methods=['GET'])
def thank_you():
    return render_template('thankyou.html')


@app.route('/manage', methods=['POST'])
def manage_post():
    req_data = request.get_json()
    selected_samples = req_data['selected_samples']
    unselected_samples = req_data['unselected_samples']
    # set database contents to these selected samples
    additions = []
    skipped_additions = []
    removals = []
    skipped_removals = []
    db = get_db()

    # Add samples (skip duplicates)
    for sample_url in selected_samples:
        try:
            db.execute('INSERT INTO samples (url) VALUES (?) ', [sample_url])
            additions.append(sample_url)
        except sqlite3.IntegrityError:
            # skip this because the sample already exists!
            skipped_additions.append(sample_url)

    # Remove samples
    for sample_url in unselected_samples:
        try:
            db.execute('DELETE FROM samples WHERE url= ?', [sample_url])
            removals.append(sample_url)
        except sqlite3.IntegrityError:
            skipped_removals.append(sample_url)

    db.commit()

    result = {'status': 'success',
              'additions': additions,
              'skipped_additions': skipped_additions,
              'removals': removals,
              'skipped_removals': skipped_removals}


    return json.dumps(result)


@app.route('/manage', methods=['GET'])
def manage_get():
    # get list of possible samples
    try:

        samples = []
        subdirs = os.listdir(SAMPLES_ROOT)

        for subdir in subdirs:
            full_subdir = os.path.join(SAMPLES_ROOT, subdir)
            if os.path.isdir(full_subdir):
                sample_names = os.listdir(full_subdir)

                for sample_name in sample_names:
                    # skip empty strings, non-files, and non mp3s
                    if not sample_name:
                        continue
                    elif not os.path.isfile(os.path.join(full_subdir, sample_name)):
                        continue
                    elif not sample_name.endswith("mp3"):
                        continue

                    sample = {
                        'url': os.path.join(SAMPLES_URL_PREFIX, subdir, sample_name),
                        'name': sample_name
                    }
                    samples.append(sample)

        db = get_db()
        samples_cur = db.execute('SELECT url FROM samples')
        entries = samples_cur.fetchall()
        db_samples = []
        for entry in entries:
            db_samples.append({
                'url': entry[0],
            })

        return render_template('manage.html', samples=json.dumps(samples), db_samples=db_samples)
    except requests.exceptions.ProxyError:
        return render_template('error.html', reason="Failed to contact sever for list of samples.")


@app.route('/wpi_participant_pool', methods=['GET'])
def wpi_participant_pool():
    return render_template('wpi_participant_pool.html')


@app.route('/', methods=['GET'])
def root():
    return redirect(url_for('interface'))


@app.route('/interface', methods=['GET'])
def interface():
    # unique ID for this labeler
    if LABELER_ID_COOKIE_KEY in request.cookies:
        labeler_id = request.cookies[LABELER_ID_COOKIE_KEY]
    else:
        labeler_id = str(uuid.uuid4())

    db = get_db()
    # get the list of samples that this labeler has already labeled
    labeled_cur = db.execute('SELECT responses.url FROM responses '
                             'JOIN samples ON samples.url = responses.url '
                             'WHERE responses.labeler_id=?', [labeler_id])
    labeled_sample_urls = [row['url'] for row in labeled_cur.fetchall()]
    all_cur = db.execute('SELECT url FROM samples')
    all_sample_urls = [row['url'] for row in all_cur.fetchall()]

    # remove any samples that have already been labeled
    unlabeled_sample_urls = [s for s in all_sample_urls if s not in labeled_sample_urls]

    if len(unlabeled_sample_urls) <= 0:
        return render_template('thankyou.html')

    # explicitly shuffle them
    np.random.shuffle(unlabeled_sample_urls)

    # Generate new UUID for this experiment. An experiment is one session by one labeler done without refreshing.
    experiment_id = str(uuid.uuid4())

    samples = [{'url': url} for url in unlabeled_sample_urls]

    href = "thankyou?"
    template = render_template('interface.html', samples=json.dumps(samples), experiment_id=experiment_id,
                               next_href=href, labeler_id=labeler_id)
    resp = make_response(template)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"

    if app.debug:
        max_age_seconds = None  # when debugging, delete cookie when browser exits
    else:
        max_age_seconds = 365 * 24 * 60 * 60  # 1 year

    resp.set_cookie(LABELER_ID_COOKIE_KEY, labeler_id, max_age=max_age_seconds)
    return resp


if __name__ == '__main__':
    app.run()
