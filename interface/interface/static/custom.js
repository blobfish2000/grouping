let audio = document.getElementById('audio');
let audio_src = document.getElementById('audio_source');
let canvas = document.getElementById('canvas');
let sample_title = document.getElementById('sample_title');
let sample_idx = 0;

/////////////////////////////////////////////////////////
// Setup
/////////////////////////////////////////////////////////

window.onload = function() {
  sample_idx = 0;
  audio_src.src = samples[sample_idx]['url'];
  sample_title.innerHTML = samples[sample_idx]['title'];
  sample_idx += 1;
  audio.load();
};

/////////////////////////////////////////////////////////
// Audio Player
/////////////////////////////////////////////////////////

function play_pause() {
  if (audio.paused) {
    audio.play();
  }
  else {
    audio.pause();
  }
}

setInterval(function() {
  // set scrubber position
  if (audio.duration > 0) {
    let x = line_begin + audio.currentTime / audio.duration *
        (line_end - line_begin);
    scrubber.x(x);
    layer.draw();
    set_time();
  }
}, 30);

function timeFmt(t) {
  return (t / 60).toFixed(0).padStart(2, '0') + ':' +
      t.toFixed(0).padStart(2, '0');
}

function set_time() {
  let now = timeFmt(audio.currentTime);
  let dur = timeFmt(audio.duration);
  let duration_string = now + ' / ' + dur;
  $('#duration').prop('innerHTML', duration_string);
}

audio.addEventListener('durationchange', function() {
  set_time();
});

audio.addEventListener('play', function() {
  let icon = $('#pause_play_button > span');
  icon.removeClass('glyphicon-play');
  icon.addClass('glyphicon-pause');
});

audio.addEventListener('suspend', function() {
  let icon = $('#pause_play_button > span');
  icon.addClass('glyphicon-play');
  icon.removeClass('glyphicon-pause');
});

audio.addEventListener('pause', function() {
  let icon = $('#pause_play_button > span');
  icon.addClass('glyphicon-play');
  icon.removeClass('glyphicon-pause');
});

function set_loop() {
  audio.loop = $('#loop_checkbox').is(':checked');
}

/////////////////////////////////////////////////////////
// Konva Canvas
/////////////////////////////////////////////////////////

function next_submit() {
  if (sample_idx === samples.length) {
    $('#next-submit-button').prop('disabled', true);
    window.location.href = 'thankyou.html';
  }
  else {
    if (sample_idx === samples.length - 1) {
      $('#next-submit-button').prop('innerHTML', 'Submit');
    }

    console.log(samples[sample_idx]);
    audio_src.src = samples[sample_idx]['url'];
    sample_title.innerHTML = samples[sample_idx]['title'];
    audio.load();

    sample_idx += 1;
  }

  reset_markers();
}

let width = 700;
let height = 100;

let stage = new Konva.Stage({
  container: 'canvas',
  width: width,
  height: height,
  fill: '#ff0',
});

let layer = new Konva.Layer();
let radius = 10;
let stroke = 2;
let line_height = 2;
let line_begin = 10;
let line_end = width - 10;

let line = new Konva.Rect({
  x: line_begin,
  y: stage.getHeight() / 2 - line_height / 2,
  width: line_end - line_begin,
  height: line_height,
  fill: '#666',
});
layer.add(line);

let background = new Konva.Rect({
  x: 0,
  y: 0,
  width: stage.getWidth(),
  height: stage.getHeight(),
  fill: '#eee',
});
layer.add(background);

let scrubber_radius = 8;
let scrubber = new Konva.Circle({
  x: line_begin,
  y: stage.getHeight() / 2,
  radius: scrubber_radius,
  fill: '#222',
  draggable: true,
  dragBoundFunc: function(pos) {
    audio.currentTime = ((pos.x - line_begin) / (line_end - line_begin) *
        audio.duration);
    return {
      x: bound(pos.x),
      y: this.getAbsolutePosition().y,
    };
  },
});

scrubber.on('mouseover', function() {
  document.body.style.cursor = 'pointer';
});

scrubber.on('mouseout', function() {
  document.body.style.cursor = 'default';
});

layer.add(scrubber);

let marker = new Konva.Circle({
  x: stage.getWidth() / 2,
  y: stage.getHeight() / 2,
  radius: radius,
  fill: '#4285F422',
  stroke: '#4285F4',
  strokeWidth: stroke,
  draggable: true,
  dragBoundFunc: function(pos) {
    return {
      x: bound(pos.x),
      y: this.getAbsolutePosition().y,
    };
  },
});

marker.on('mouseover', function() {
  document.body.style.cursor = 'pointer';
});

marker.on('mouseout', function() {
  document.body.style.cursor = 'default';
});

marker.on('click', function(event) {
  if (event.evt.shiftKey) {
    this.destroy();
    layer.draw();
  }
});

layer.on('click', function(event) {
  if (event.evt.ctrlKey) {
    // insert new marker
    let x = stage.getPointerPosition().x;
    if ((x > line_begin) &&
        (x < line_end)) {
      add_marker(x);
    }
  } else {
    // do nothing on normal click
  }
});

background.setZIndex(0);
line.setZIndex(1);
add_marker((line_end + line_begin) / 2);
stage.add(layer);
let new_markers = [];

function bound(x) {
  return Math.max(line_begin,
      Math.min(line_end, x));
}

function add_marker(x_pos) {
  let clone = marker.clone({
    x: x_pos,
    y: stage.getHeight() / 2,
  });
  layer.add(clone);
  layer.draw();
}

function reset_markers() {
}
