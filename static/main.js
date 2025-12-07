const statusEl = document.getElementById('status');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const sourceEl = document.getElementById('source');
const tzEl = document.getElementById('tz');
const historyTbody = document.getElementById('history');
const downloadBtn = document.getElementById('downloadBtn');
const monitorView = document.getElementById('monitorView');
const analyticsView = document.getElementById('analyticsView');
const tabButtons = document.querySelectorAll('.tab-button');
const sumTotalEvents = document.getElementById('sumTotalEvents');
const sumByLabelEl = document.getElementById('sumByLabel');
const sumTimeSpan = document.getElementById('sumTimeSpan');
const sumAvgScore = document.getElementById('sumAvgScore');
const sumLastEvent = document.getElementById('sumLastEvent');
const labelFilters = document.querySelectorAll('.label-filter');
const timelineEl = document.getElementById('timeline');


const MAX_VISUAL_EVENTS = 150;

let evtSrc = null;
let analyticsInitialized = false;
let chart = null;
let chartData = [];
let runHeader = null;
let activeTimelineId = null;
let statusInterval = null;
let thumbnailRefreshInterval = null;

function setStatus(text) {
  statusEl.textContent = text;
}

function addEventRow(ev) {
  const tr = document.createElement('tr');
  const tdWhen = document.createElement('td');
  const tdLabel = document.createElement('td');
  const tdScore = document.createElement('td');
  const tdSnap = document.createElement('td');
  tdWhen.textContent = ev.time_iso || '';
  tdLabel.textContent = ev.label || (ev.type === 'run_start' ? 'run_start' : '');
  tdScore.textContent = typeof ev.score === 'number' ? ev.score.toFixed(3) : '';
  if (ev.time && ev.label && typeof ev.score === 'number') {
    const t = Math.trunc(ev.time);
    const name = `${t}_${ev.label}_${ev.score.toFixed(2)}.jpg`;
    const a = document.createElement('a');
    a.href = `/snapshots/${name}`;
    a.textContent = 'snapshot';
    a.target = '_blank';
    tdSnap.appendChild(a);
  }
  tr.appendChild(tdWhen);
  tr.appendChild(tdLabel);
  tr.appendChild(tdScore);
  tr.appendChild(tdSnap);
  historyTbody.appendChild(tr);
}

function labelColor(label) {
  if (label === 'lighting_change') return '#f5c430';
  if (label === 'camera_motion') return '#509eff';
  if (label === 'scene_or_object_change') return '#f472b6';
  return '#9aa0a6';
}

function recomputeSummary() {
  const events = chartData;
  sumTotalEvents.textContent = events.length.toString();

  const counts = {};
  let minT = Infinity;
  let maxT = -Infinity;
  let sumScore = 0;
  let lastEvent = null;

  for (const ev of events) {
    const label = ev.label || 'unknown';
    counts[label] = (counts[label] || 0) + 1;
    if (typeof ev.time === 'number') {
      minT = Math.min(minT, ev.time);
      maxT = Math.max(maxT, ev.time);
    }
    if (typeof ev.score === 'number') {
      sumScore += ev.score;
    }
    if (!lastEvent || (typeof ev.time === 'number' && ev.time > lastEvent.time)) {
      lastEvent = ev;
    }
  }

  sumByLabelEl.innerHTML = '';
  Object.entries(counts).forEach(([label, count]) => {
    const chip = document.createElement('span');
    chip.className = 'chip chip-muted';
    chip.textContent = `${label} · ${count}`;
    sumByLabelEl.appendChild(chip);
  });

  if (!isFinite(minT) || !isFinite(maxT) || events.length === 0) {
    sumTimeSpan.textContent = '–';
  } else {
    const start = new Date(minT * 1000);
    const end = new Date(maxT * 1000);
    sumTimeSpan.textContent = `${start.toLocaleTimeString()} → ${end.toLocaleTimeString()}`;
  }

  if (events.length === 0) {
    sumAvgScore.textContent = '–';
  } else {
    sumAvgScore.textContent = (sumScore / events.length).toFixed(3);
  }

  if (!lastEvent) {
    sumLastEvent.textContent = '–';
  } else {
    const when = lastEvent.time_iso || new Date(lastEvent.time * 1000).toLocaleTimeString();
    sumLastEvent.textContent = `${lastEvent.label || 'event'} @ ${when}`;
  }
}

function buildChart() {
  const canvas = document.getElementById('scoreChart');
  if (!canvas) {
    console.warn('scoreChart canvas not found');
    return;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    console.warn('Could not get 2d context');
    return;
  }
  
  // Destroy existing chart if present
  if (chart) {
    chart.destroy();
    chart = null;
  }
  
  chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Events',
          data: [],
          pointRadius: 5,
          pointHoverRadius: 7,
          pointBackgroundColor: '#9aa0a6',
          pointBorderColor: '#9aa0a6',
          showLine: false,
        },
        {
          label: 'Threshold',
          data: [],
          type: 'line',
          borderColor: '#5f6368',
          borderDash: [5, 5],
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: 'index',
      },
      scales: {
        x: {
          type: 'linear',
          position: 'bottom',
          ticks: {
            color: '#9aa0a6',
            font: { size: 11 },
            callback(value) {
              if (typeof value !== 'number' || !isFinite(value)) return '';
              const d = new Date(value * 1000);
              return d.toLocaleTimeString();
            },
          },
          grid: { 
            color: '#1d2128',
            drawBorder: true,
            borderColor: '#2b3038',
          },
          title: {
            display: true,
            text: 'Time',
            color: '#9aa0a6',
            font: { size: 12 },
          },
        },
        y: {
          ticks: { 
            color: '#9aa0a6',
            font: { size: 11 },
          },
          grid: { 
            color: '#1d2128',
            drawBorder: true,
            borderColor: '#2b3038',
          },
          title: {
            display: true,
            text: 'Score',
            color: '#9aa0a6',
            font: { size: 12 },
          },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: { 
            color: '#9aa0a6',
            font: { size: 11 },
            usePointStyle: true,
          },
        },
        tooltip: {
          enabled: true,
          backgroundColor: 'rgba(15, 18, 22, 0.95)',
          titleColor: '#e8eaed',
          bodyColor: '#9aa0a6',
          borderColor: '#2b3038',
          borderWidth: 1,
          padding: 10,
          callbacks: {
            title(context) {
              const ev = context[0].raw && context[0].raw._event;
              if (ev && ev.time_iso) return ev.time_iso;
              if (ev && typeof ev.time === 'number') {
                return new Date(ev.time * 1000).toLocaleString();
              }
              return 'Event';
            },
            label(context) {
              const ev = context.raw && context.raw._event;
              if (!ev) return `Score: ${context.parsed.y.toFixed(3)}`;
              const comps = ev.components || {};
              const tooltipLines = [
                `Score: ${ev.score.toFixed(3)}`,
                `Label: ${ev.label || 'unknown'}`,
                `hist: ${Number(comps.hist || 0).toFixed(3)}`,
                `bright: ${Number(comps.brightness || 0).toFixed(3)}`,
                `edge: ${Number(comps.edge || 0).toFixed(3)}`,
                `ssim: ${Number(comps.ssim || 0).toFixed(3)}`,
                `cnn: ${Number(comps.cnn || 0).toFixed(3)}`,
              ];
              if (comps.mmd !== undefined) {
                tooltipLines.push(`mmd: ${Number(comps.mmd || 0).toFixed(3)}`);
              }
              if (comps.mahalanobis !== undefined) {
                tooltipLines.push(`mahalanobis: ${Number(comps.mahalanobis || 0).toFixed(3)}`);
              }
              return tooltipLines;
            },
          },
        },
      },
      onClick(evt, elements) {
        if (!elements.length) return;
        const element = elements[0];
        const datasetIndex = element.datasetIndex;
        const index = element.index;
        if (datasetIndex === 0 && chart.data.datasets[0].data[index]) {
          const ev = chart.data.datasets[0].data[index]._event;
          if (ev && ev._id) {
            setActiveTimelineItem(ev._id, true);
          }
        }
      },
    },
  });
}

function updateChart() {
  if (!chart) return;
  const selectedLabels = Array.from(labelFilters)
    .filter((cb) => cb.checked)
    .map((cb) => cb.value);

  const points = [];
  const thresholdPoints = [];
  const threshold = Number(runHeader && runHeader.threshold) || 0.08;
  
  for (const ev of chartData) {
    if (!ev.label || !selectedLabels.includes(ev.label)) continue;
    if (typeof ev.time !== 'number' || !isFinite(ev.time)) continue;
    if (typeof ev.score !== 'number' || !isFinite(ev.score)) continue;
    
    const x = ev.time;
    const y = ev.score;
    const color = labelColor(ev.label);
    
    points.push({
      x,
      y,
      _event: ev,
    });
    
    thresholdPoints.push({
      x,
      y: Number(ev.threshold || threshold),
    });
  }

  // Set point colors per point for scatter chart
  // Update scatter points with colors per point
  chart.data.datasets[0].data = points;
  chart.data.datasets[0].pointBackgroundColor = points.map(pt => labelColor(pt._event.label));
  chart.data.datasets[0].pointBorderColor = points.map(pt => labelColor(pt._event.label));
  
  // Update threshold line
  chart.data.datasets[1].data = thresholdPoints;
  
  // Update chart with animation disabled for performance
  chart.update('none');
}

function refreshTimelineThumbnails() {
  // Periodically refresh thumbnails that failed to load
  const timelineItems = timelineEl.querySelectorAll('.timeline-item');
  let refreshed = false;
  
  timelineItems.forEach(item => {
    const placeholder = item.querySelector('.timeline-thumb-placeholder');
    const existingImg = item.querySelector('.timeline-thumb');
    
    if (placeholder || (existingImg && !existingImg.complete && existingImg.naturalWidth === 0)) {
      // Try to load image again for items that showed placeholder or failed
      const ev = chartData.find(e => e._id === item.dataset.id);
      if (ev) {
        // Try snapshot_url first, then construct from event data
        let snapshotUrl = ev.snapshot_url;
        if (!snapshotUrl && ev.time && ev.label && ev.score !== undefined) {
          const t = Math.trunc(ev.time);
          const name = `${t}_${ev.label}_${ev.score.toFixed(2)}.jpg`;
          snapshotUrl = `/snapshots/${name}`;
        }
        
        if (snapshotUrl) {
          const container = item.querySelector('.timeline-thumb-container');
          if (container) {
            const img = new Image();
            img.className = 'timeline-thumb';
            img.alt = ev.label || 'snapshot';
            
            img.onload = function() {
              // Remove placeholder and failed images
              if (placeholder) placeholder.remove();
              if (existingImg && !existingImg.complete) existingImg.remove();
              // Add new image if not already present
              if (!container.querySelector('.timeline-thumb')) {
                container.appendChild(this);
              }
              refreshed = true;
            };
            
            img.onerror = function() {
              // Keep placeholder if image still fails
              if (!placeholder && !container.querySelector('.timeline-thumb-placeholder')) {
                const newPlaceholder = document.createElement('div');
                newPlaceholder.className = 'timeline-thumb-placeholder';
                newPlaceholder.textContent = 'No image';
                container.appendChild(newPlaceholder);
              }
            };
            
            img.src = snapshotUrl + '?t=' + Date.now();
          }
        }
      }
    }
  });
  
  return refreshed;
}

function buildTimeline() {
  timelineEl.innerHTML = '';
  const selectedLabels = Array.from(labelFilters)
    .filter((cb) => cb.checked)
    .map((cb) => cb.value);

  for (const ev of chartData) {
    if (!selectedLabels.includes(ev.label)) continue;
    const id = ev._id;
    const item = document.createElement('div');
    item.className = 'timeline-item';
    item.dataset.id = id;

    // Always create a thumbnail container, even if snapshot_url is missing
    const thumbContainer = document.createElement('div');
    thumbContainer.className = 'timeline-thumb-container';
    
    // Determine snapshot URL
    let snapshotUrl = ev.snapshot_url;
    if (!snapshotUrl && ev.time && ev.label && ev.score !== undefined) {
      const t = Math.trunc(ev.time);
      const name = `${t}_${ev.label}_${ev.score.toFixed(2)}.jpg`;
      snapshotUrl = `/snapshots/${name}`;
    }
    
    if (snapshotUrl) {
      const img = document.createElement('img');
      img.src = snapshotUrl + '?t=' + Date.now(); // Cache busting
      img.alt = ev.label || 'snapshot';
      img.className = 'timeline-thumb';
      
      // Handle image loading errors with retry logic
      let retryCount = 0;
      const maxRetries = 3;
      
      img.onerror = function() {
        retryCount++;
        if (retryCount < maxRetries) {
          // Retry with cache busting
          setTimeout(() => {
            this.src = snapshotUrl + '?t=' + Date.now() + retryCount;
          }, 500 * retryCount); // Stagger retries
          return;
        }
        // If still fails after retries, show placeholder
        this.style.display = 'none';
        if (!thumbContainer.querySelector('.timeline-thumb-placeholder')) {
          const placeholder = document.createElement('div');
          placeholder.className = 'timeline-thumb-placeholder';
          placeholder.textContent = 'Loading...';
          thumbContainer.appendChild(placeholder);
        }
      };
      
      img.onload = function() {
        // Remove placeholder if image loads successfully
        const placeholder = thumbContainer.querySelector('.timeline-thumb-placeholder');
        if (placeholder) {
          placeholder.remove();
        }
      };
      
      thumbContainer.appendChild(img);
    } else {
      thumbContainer.innerHTML = '<div class="timeline-thumb-placeholder">No image</div>';
    }
    
    item.appendChild(thumbContainer);

    const topRow = document.createElement('div');
    const badge = document.createElement('span');
    badge.className = 'badge ' + (
      ev.label === 'lighting_change' ? 'badge-lighting' :
      ev.label === 'camera_motion' ? 'badge-camera' :
      'badge-scene'
    );
    badge.textContent = ev.label || 'event';
    topRow.appendChild(badge);
    item.appendChild(topRow);

    const meta = document.createElement('div');
    meta.className = 'timeline-meta';
    const when = document.createElement('span');
    when.textContent = ev.time_iso || '';
    const score = document.createElement('span');
    score.textContent = typeof ev.score === 'number' ? ev.score.toFixed(3) : '';
    meta.appendChild(when);
    meta.appendChild(score);
    item.appendChild(meta);

    item.addEventListener('click', () => {
      setActiveTimelineItem(id, true);
    });

    timelineEl.appendChild(item);
  }
  
  // Refresh thumbnails after building with staggered delays
  setTimeout(refreshTimelineThumbnails, 1000);
  setTimeout(refreshTimelineThumbnails, 3000);
}

function setActiveTimelineItem(id, scrollIntoView) {
  activeTimelineId = id;
  const items = timelineEl.querySelectorAll('.timeline-item');
  items.forEach((el) => {
    el.classList.toggle('active', el.dataset.id === id);
  });
  if (scrollIntoView) {
    const el = timelineEl.querySelector(`.timeline-item[data-id="${id}"]`);
    if (el) el.scrollIntoView({ behavior: 'smooth', inline: 'center' });
  }
}

async function loadAnalyticsData() {
  try {
    const resp = await fetch('/events.json');
    const data = await resp.json();
    if (!data.ok) return;
    runHeader = data.run || null;
    const allEvents = data.events || [];
    // Keep only the most recent MAX_VISUAL_EVENTS events.
    const sliceStart = Math.max(0, allEvents.length - MAX_VISUAL_EVENTS);
    chartData = [];
    let counter = 0;
    for (let i = sliceStart; i < allEvents.length; i++) {
      const ev = allEvents[i];
      const copy = Object.assign({}, ev);
      copy._id = `ev-${counter++}`;
      chartData.push(copy);
    }
    recomputeSummary();
    
    // Ensure Chart.js is loaded before building
    if (typeof Chart === 'undefined') {
      console.warn('Chart.js not loaded yet, retrying...');
      setTimeout(() => {
        if (typeof Chart !== 'undefined' && !chart) {
          buildChart();
          updateChart();
          buildTimeline();
        }
      }, 100);
      return;
    }
    
    if (!chart) {
      buildChart();
    }
    updateChart();
    buildTimeline();
  } catch (e) {
    console.error('Failed to load analytics data', e);
  }
}

function ensureAnalyticsInitialized() {
  if (analyticsInitialized) return;
  analyticsInitialized = true;
  
  // Wait a bit to ensure Chart.js is loaded
  if (typeof Chart === 'undefined') {
    setTimeout(() => {
      loadAnalyticsData();
    }, 200);
  } else {
    loadAnalyticsData();
  }
}

labelFilters.forEach((cb) => {
  cb.addEventListener('change', () => {
    updateChart();
    buildTimeline();
  });
});

timelineEl.addEventListener('keydown', (e) => {
  if (!chartData.length) return;
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  e.preventDefault();
  const ids = Array.from(timelineEl.querySelectorAll('.timeline-item')).map((el) => el.dataset.id);
  if (!ids.length) return;
  let idx = activeTimelineId ? ids.indexOf(activeTimelineId) : -1;
  if (e.key === 'ArrowRight') {
    idx = Math.min(ids.length - 1, idx + 1);
  } else if (e.key === 'ArrowLeft') {
    idx = Math.max(0, idx - 1);
  }
  if (idx >= 0 && idx < ids.length) {
    setActiveTimelineItem(ids[idx], true);
  }
});

tabButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    tabButtons.forEach((b) => b.classList.toggle('active', b === btn));
    if (tab === 'monitor') {
      monitorView.classList.remove('hidden');
      analyticsView.classList.add('hidden');
    } else {
      monitorView.classList.add('hidden');
      analyticsView.classList.remove('hidden');
      ensureAnalyticsInitialized();
    }
  });
});

async function startRun() {
  const source = sourceEl.value.trim();
  const tz = tzEl.value.trim();
  if (!source) {
    alert('Please enter a link');
    return;
  }
  setStatus('Starting...');
  const resp = await fetch('/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source, tz })
  });
  const j = await resp.json();
  if (!j.ok) {
    setStatus('Error: ' + (j.error || 'start failed'));
    return;
  }
  startBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus('Running');
  historyTbody.innerHTML = '';
  chartData = [];
  runHeader = null;
  
  // Initialize chart if Analytics tab is visible and Chart.js is loaded
  if (!analyticsView.classList.contains('hidden')) {
    ensureAnalyticsInitialized();
  }
  
  if (chart) {
    chart.data.datasets[0].data = [];
    chart.data.datasets[1].data = [];
    chart.update('none');
  }
  timelineEl.innerHTML = '';
  activeTimelineId = null;

  if (evtSrc) evtSrc.close();
  evtSrc = new EventSource('/events');
  evtSrc.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.type === 'run_start') {
        runHeader = data;
        addEventRow(data);
      } else if (data.type === 'video_info') {
        updateVideoStats(data);
      } else if (data.type === 'error') {
        setStatus('Error: ' + data.message);
        if (statusInterval) {
          clearInterval(statusInterval);
          statusInterval = null;
        }
      } else if (data.label) {
        addEventRow(data);
        const copy = Object.assign({}, data);
        copy._id = `live-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        chartData.push(copy);
        // Trim to most recent MAX_VISUAL_EVENTS to avoid unbounded growth.
        if (chartData.length > MAX_VISUAL_EVENTS) {
          chartData.splice(0, chartData.length - MAX_VISUAL_EVENTS);
        }
        recomputeSummary();
        updateChart();
        buildTimeline();
        
        // Retry loading thumbnails after a short delay (snapshot might still be saving)
        setTimeout(() => {
          const timelineItems = timelineEl.querySelectorAll('.timeline-item');
          timelineItems.forEach(item => {
            const img = item.querySelector('.timeline-thumb');
            if (img && !img.complete) {
              const currentSrc = img.src;
              img.src = '';
              setTimeout(() => {
                img.src = currentSrc;
              }, 100);
            }
          });
        }, 500);
      }
    } catch (_) {
      // ignore pings
    }
  };
  evtSrc.onerror = () => {
    setStatus('Event stream error');
    if (statusInterval) {
      clearInterval(statusInterval);
      statusInterval = null;
    }
  };
  
  // Start polling for status updates
  if (statusInterval) clearInterval(statusInterval);
  statusInterval = setInterval(updateStatus, 500);
  updateStatus();
  
  // Start periodic thumbnail refresh (every 5 seconds)
  if (thumbnailRefreshInterval) clearInterval(thumbnailRefreshInterval);
  thumbnailRefreshInterval = setInterval(refreshTimelineThumbnails, 5000);
}

async function stopRun() {
  setStatus('Stopping...');
  await fetch('/stop', { method: 'POST' });
  if (evtSrc) evtSrc.close();
  evtSrc = null;
  if (statusInterval) {
    clearInterval(statusInterval);
    statusInterval = null;
  }
  if (thumbnailRefreshInterval) {
    clearInterval(thumbnailRefreshInterval);
    thumbnailRefreshInterval = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus('Stopped');
  resetVideoStats();
}

function updateVideoStats(data) {
  const fpsEl = document.getElementById('statFPS');
  const progressEl = document.getElementById('statProgress');
  const framesEl = document.getElementById('statFrames');
  
  if (fpsEl && data.fps) {
    fpsEl.textContent = data.fps.toFixed(2);
  }
  if (framesEl && data.frame_count) {
    framesEl.textContent = `${data.current_frame || 0} / ${data.frame_count}`;
  }
  if (progressEl && data.duration) {
    const currentTime = (data.current_frame || 0) / (data.fps || 1);
    const progress = data.frame_count ? ((data.current_frame || 0) / data.frame_count * 100) : 0;
    progressEl.textContent = `${currentTime.toFixed(1)}s / ${data.duration.toFixed(1)}s (${progress.toFixed(1)}%)`;
  }
}

async function updateStatus() {
  try {
    const resp = await fetch('/status');
    const data = await resp.json();
    if (!data.ok) return;
    
    const statusEl = document.getElementById('statStatus');
    const processingFpsEl = document.getElementById('statProcessingFPS');
    const progressEl = document.getElementById('statProgress');
    const framesEl = document.getElementById('statFrames');
    
    if (statusEl) {
      statusEl.textContent = data.running ? (data.is_live ? 'Live' : 'Playing') : 'Stopped';
    }
    
    if (processingFpsEl && data.processing_fps) {
      processingFpsEl.textContent = data.processing_fps.toFixed(1) + ' FPS';
    }
    
    if (data.running && !data.is_live) {
      if (progressEl && data.progress !== null) {
        progressEl.textContent = `${data.progress.toFixed(1)}%`;
      }
      if (framesEl && data.frame_count) {
        framesEl.textContent = `${data.current_frame || 0} / ${data.frame_count}`;
      }
    }
  } catch (e) {
    // Ignore errors
  }
}

function resetVideoStats() {
  document.getElementById('statStatus').textContent = 'Idle';
  document.getElementById('statFPS').textContent = '–';
  document.getElementById('statProcessingFPS').textContent = '–';
  document.getElementById('statProgress').textContent = '–';
  document.getElementById('statFrames').textContent = '–';
}

startBtn.addEventListener('click', startRun);
stopBtn.addEventListener('click', stopRun);
downloadBtn.addEventListener('click', () => {
  window.location.href = '/download.csv';
});
