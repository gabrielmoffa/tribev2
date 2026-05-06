const runsCache = { runs: [], loaded: false };
const panels = [];
const globalState = { active3dPanel: null };

const panelsContainer = document.querySelector("#panels");
const panelTemplate = document.querySelector("#panelTemplate");
const globalMeta = document.querySelector("#globalMeta");

function formatTime(seconds) {
  return `${seconds.toFixed(1)}s`;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function waitForMetadata(video) {
  if (Number.isFinite(video.duration) && video.duration > 0) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    video.addEventListener("loadedmetadata", resolve, { once: true });
    video.addEventListener("error", resolve, { once: true });
  });
}

function multiplyMatrix(a, b) {
  return [
    [
      a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
      a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
      a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
    ],
    [
      a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
      a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
      a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
    ],
    [
      a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
      a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
      a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
    ],
  ];
}

function rotationXMatrix(angle) {
  const s = Math.sin(angle);
  const c = Math.cos(angle);
  return [
    [1, 0, 0],
    [0, c, -s],
    [0, s, c],
  ];
}

function rotationYMatrix(angle) {
  const s = Math.sin(angle);
  const c = Math.cos(angle);
  return [
    [c, 0, -s],
    [0, 1, 0],
    [s, 0, c],
  ];
}

function rotationZMatrix(angle) {
  const s = Math.sin(angle);
  const c = Math.cos(angle);
  return [
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1],
  ];
}

function rotationMatrixForAxis(axis, angle) {
  if (axis === "x") return rotationXMatrix(angle);
  if (axis === "y") return rotationYMatrix(angle);
  return rotationZMatrix(angle);
}

function eulerMatrix(x, y) {
  return multiplyMatrix(rotationXMatrix(x), rotationYMatrix(y));
}

function applyMatrix(matrix, x, y, z) {
  return [
    matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
    matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
    matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
  ];
}

function mix(a, b, t) {
  return Math.round(a + (b - a) * t);
}

class ReviewPanel {
  constructor() {
    this.root = panelTemplate.content.firstElementChild.cloneNode(true);
    this.state = {
      data: null,
      duration: 0,
      dragging: false,
      playing: false,
      brainMode: "video",
      mesh: null,
      frameCache: new Map(),
      brainFrameSecond: null,
      currentFrame: null,
      rotationMatrix: eulerMatrix(0, 0),
      zoom: 1.2,
      brainDragging: false,
      brainPointer: null,
      activeRing: null,
      activeRingTangent: null,
      ringPaths: [],
      selectedSeries: new Set(),
      syncingNativeControls: false,
      functionSource: "anatomy",
    };
    this._cacheElements();
    this._populateRunSelect();
    this._bindEvents();
    panelsContainer.appendChild(this.root);
  }

  _q(sel) { return this.root.querySelector(sel); }
  _qa(sel) { return this.root.querySelectorAll(sel); }

  _cacheElements() {
    this.runSelect = this._q(".runSelect");
    this.playButton = this._q(".playButton");
    this.timeReadout = this._q(".timeReadout");
    this.sourceVideo = this._q(".sourceVideo");
    this.brainVideo = this._q(".brainVideo");
    this.brainVideoTab = this._q(".brainVideoTab");
    this.brain3dTab = this._q(".brain3dTab");
    this.brain3dPane = this._q(".brain3dPane");
    this.brainCanvas = this._q(".brainCanvas");
    this.brain3dStatus = this._q(".brainStatus");
    this.chartCanvas = this._q(".chartCanvas");
    this.scrubber = this._q(".scrubber");
    this.legend = this._q(".legend");
    this.panelMeta = this._q(".panelMeta");
    this.panelTitle = this._q(".panelTitle");
    this.sourceLabel = this._q(".sourceLabel");
    this.selectedSecond = this._q(".selectedSecond");
    this.activationPanelMeta = this._q(".activationPanelMeta");
    this.activeFunctions = this._q(".activeFunctions");
    this.removeButton = this._q(".removePanelButton");
    this.viewButtons = this._qa(".viewButton");
    this.ctx = this.chartCanvas.getContext("2d");
    this.brainCtx = this.brainCanvas.getContext("2d");
  }

  _populateRunSelect(selectedId) {
    this.runSelect.innerHTML = "";
    for (const run of runsCache.runs) {
      const option = document.createElement("option");
      option.value = run.id;
      option.textContent = `${run.name} (${run.seconds}s)`;
      this.runSelect.appendChild(option);
    }
    if (selectedId) this.runSelect.value = selectedId;
  }

  _bindEvents() {
    this.playButton.addEventListener("click", () => {
      if (this.state.playing) this.pause();
      else this.play();
    });

    this.brainVideoTab.addEventListener("click", () => this.setBrainMode("video"));
    this.brain3dTab.addEventListener("click", () => this.setBrainMode("3d"));

    this._qa(".sourceToggle .tabButton").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.state.functionSource = btn.dataset.source;
        this._qa(".sourceToggle .tabButton").forEach((sib) => {
          const active = sib === btn;
          sib.classList.toggle("active", active);
          sib.setAttribute("aria-selected", String(active));
        });
        const sec = Math.round(Number(this.scrubber.value) || 0);
        this.renderActiveFunctions(sec);
      });
    });

    this.runSelect.addEventListener("change", () => this.loadRun(this.runSelect.value).then(persistOpenPanels));

    this.scrubber.addEventListener("input", (e) => this.seekTo(e.target.value));

    this.chartCanvas.addEventListener("pointerdown", (e) => {
      this.state.dragging = true;
      this.chartCanvas.setPointerCapture(e.pointerId);
      this._seekFromPointer(e);
    });
    this.chartCanvas.addEventListener("pointermove", (e) => {
      if (this.state.dragging) this._seekFromPointer(e);
    });
    this.chartCanvas.addEventListener("pointerup", () => { this.state.dragging = false; });

    this.sourceVideo.addEventListener("ended", () => this.pause());
    this.sourceVideo.addEventListener("play", () => this._handleNativePlay());
    this.brainVideo.addEventListener("play", () => this._handleNativePlay());
    this.sourceVideo.addEventListener("pause", (e) => this._handleNativePause(e));
    this.brainVideo.addEventListener("pause", (e) => this._handleNativePause(e));
    this.sourceVideo.addEventListener("timeupdate", () => this._handleSourceTimelineChange());
    this.sourceVideo.addEventListener("seeked", () => this._handleSourceTimelineChange());
    this.brainVideo.addEventListener("seeked", () => this._handleBrainSeek());

    this.brainCanvas.addEventListener("pointerdown", (e) => this._brainPointerDown(e));
    this.brainCanvas.addEventListener("pointermove", (e) => this._brainPointerMove(e));
    this.brainCanvas.addEventListener("pointerup", (e) => this._brainPointerUp(e));
    this.brainCanvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.state.zoom = Math.max(0.72, Math.min(2.2, this.state.zoom - e.deltaY * 0.001));
      this.drawBrain();
    });
    this.viewButtons.forEach((btn) => {
      btn.addEventListener("click", () => this.setBrainView(btn.dataset.view));
    });

    this.removeButton.addEventListener("click", () => this.remove());
  }

  remove() {
    this.pause();
    if (globalState.active3dPanel === this) globalState.active3dPanel = null;
    const idx = panels.indexOf(this);
    if (idx >= 0) panels.splice(idx, 1);
    this.root.remove();
    persistOpenPanels();
  }

  async loadRun(runId) {
    this.pause();
    const run = runsCache.runs.find((item) => item.id === runId);
    if (!run) return;
    this.runSelect.value = runId;
    this.state.data = await fetchJson(`/api/runs/${runId}/data`);
    this.state.duration = Math.max(0, this.state.data.seconds.length - 1);
    this.sourceVideo.src = run.inputVideo;
    this.brainVideo.src = run.brainVideo;
    this.sourceVideo.load();
    this.brainVideo.load();
    await Promise.allSettled([waitForMetadata(this.sourceVideo), waitForMetadata(this.brainVideo)]);
    const sourceDuration = Number.isFinite(this.sourceVideo.duration) ? this.sourceVideo.duration : Infinity;
    const brainDuration = Number.isFinite(this.brainVideo.duration) ? this.brainVideo.duration : Infinity;
    this.state.duration = Math.min(this.state.duration, sourceDuration, brainDuration);
    this.state.frameCache = new Map();
    this.state.brainFrameSecond = null;
    this.state.mesh = null;
    this.state.selectedSeries = new Set(
      this.state.data.series.filter((s) => s.selected).map((s) => s.key),
    );
    this.scrubber.max = String(this.state.duration);
    this.scrubber.value = "0";
    this.panelTitle.textContent = this.state.data.run.name;
    this.panelMeta.textContent = `${this.state.data.shape[0]} seconds, ${this.state.data.shape[1].toLocaleString()} cortical vertices`;
    this.sourceLabel.textContent = this.state.data.run.path;
    this.renderLegend();
    this.seekTo(0.01);
    window.setTimeout(() => this.seekTo(0), 150);
    this.drawChart();
    if (this.state.brainMode === "3d") {
      await this.ensureInteractiveBrain();
    }
  }

  renderLegend() {
    this.legend.innerHTML = "";
    for (const series of this.state.data.series) {
      const item = document.createElement("button");
      item.className = "legendItem";
      item.type = "button";
      item.dataset.key = series.key;
      item.classList.toggle("inactive", !this.state.selectedSeries.has(series.key));
      item.title = [
        `${series.name} | peak ${series.peak.toFixed(3)} | ${series.vertices} vertices`,
        series.function?.summary,
        series.function?.lateralization,
      ].filter(Boolean).join("\n\n");
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = series.color;
      item.append(swatch, document.createTextNode(series.name));
      item.addEventListener("click", () => {
        if (this.state.selectedSeries.has(series.key)) {
          this.state.selectedSeries.delete(series.key);
        } else {
          this.state.selectedSeries.add(series.key);
        }
        item.classList.toggle("inactive", !this.state.selectedSeries.has(series.key));
        this.drawChart();
      });
      this.legend.appendChild(item);
    }
  }

  _chartBounds() {
    const ratio = window.devicePixelRatio || 1;
    const cssWidth = this.chartCanvas.clientWidth;
    const cssHeight = this.chartCanvas.clientHeight;
    if (this.chartCanvas.width !== Math.round(cssWidth * ratio)) {
      this.chartCanvas.width = Math.round(cssWidth * ratio);
      this.chartCanvas.height = Math.round(cssHeight * ratio);
    }
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { width: cssWidth, height: cssHeight, left: 56, right: 18, top: 16, bottom: 34 };
  }

  _allValues() {
    if (!this.state.data) return [0, 1];
    const selected = this.state.data.series.filter((s) => this.state.selectedSeries.has(s.key));
    return selected.flatMap((s) => s.values);
  }

  drawChart() {
    if (!this.state.data) return;
    const ctx = this.ctx;
    const b = this._chartBounds();
    const plotW = b.width - b.left - b.right;
    const plotH = b.height - b.top - b.bottom;
    const maxValue = Math.max(...this._allValues(), ...this.state.data.overall, 0.001);
    const current = Number(this.scrubber.value);

    ctx.clearRect(0, 0, b.width, b.height);
    ctx.fillStyle = "#101316";
    ctx.fillRect(0, 0, b.width, b.height);

    ctx.strokeStyle = "#27303a";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#9faab5";
    ctx.font = "12px system-ui";
    for (let i = 0; i <= 4; i++) {
      const y = b.top + (plotH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(b.left, y);
      ctx.lineTo(b.left + plotW, y);
      ctx.stroke();
      const label = ((maxValue * (4 - i)) / 4).toFixed(2);
      ctx.fillText(label, 10, y + 4);
    }

    for (let s = 0; s <= this.state.duration; s++) {
      const x = b.left + (s / Math.max(this.state.duration, 1)) * plotW;
      ctx.strokeStyle = "#20272f";
      ctx.beginPath();
      ctx.moveTo(x, b.top);
      ctx.lineTo(x, b.top + plotH);
      ctx.stroke();
      ctx.fillStyle = "#7e8993";
      ctx.fillText(`${s}s`, x - 7, b.top + plotH + 24);
    }

    for (const series of this.state.data.series.filter((it) => this.state.selectedSeries.has(it.key))) {
      this._drawLine(series.values, series.color, 2.3, maxValue, b, plotW, plotH);
    }
    this._drawLine(this.state.data.overall, "#f4f0e8", 1.6, maxValue, b, plotW, plotH, [5, 5]);

    const playheadX = b.left + (current / Math.max(this.state.duration, 1)) * plotW;
    ctx.strokeStyle = "#e9b44c";
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(playheadX, b.top - 2);
    ctx.lineTo(playheadX, b.top + plotH + 4);
    ctx.stroke();

    const index = Math.max(0, Math.min(this.state.data.seconds.length - 1, Math.round(current)));
    this.selectedSecond.textContent = `Second ${index}`;
    this.renderActiveFunctions(index);
  }

  _drawLine(values, color, width, maxValue, b, plotW, plotH, dash = []) {
    const ctx = this.ctx;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    values.forEach((value, index) => {
      const x = b.left + (index / Math.max(this.state.duration, 1)) * plotW;
      const y = b.top + plotH - (value / maxValue) * plotH;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  _seekFromPointer(event) {
    const b = this._chartBounds();
    const rect = this.chartCanvas.getBoundingClientRect();
    const x = Math.min(Math.max(event.clientX - rect.left - b.left, 0), b.width - b.left - b.right);
    const seconds = (x / Math.max(b.width - b.left - b.right, 1)) * this.state.duration;
    this.seekTo(seconds);
  }

  seekTo(seconds) {
    const clamped = Math.max(0, Math.min(Number(seconds) || 0, this.state.duration));
    this.state.syncingNativeControls = true;
    this.sourceVideo.currentTime = clamped;
    this.brainVideo.currentTime = clamped;
    this.state.syncingNativeControls = false;
    this.renderAtTime(clamped);
  }

  renderAtTime(seconds) {
    const clamped = Math.max(0, Math.min(Number(seconds) || 0, this.state.duration));
    this.scrubber.value = String(clamped);
    this.timeReadout.textContent = formatTime(clamped);
    this.drawChart();
    this.updateInteractiveBrain();
  }

  renderActiveFunctions(second) {
    if (!this.state.data) return;
    const active = this.state.data.series
      .map((series) => ({
        ...series,
        currentValue: series.values[second] || 0,
        signedValue: series.signed?.[second] || 0,
      }))
      .filter((series) => series.currentValue > 0)
      .sort((a, b) => b.currentValue - a.currentValue)
      .slice(0, 6);

    this.activationPanelMeta.textContent = `Top regions at second ${second}`;
    this.activeFunctions.innerHTML = "";
    for (const series of active) {
      const item = document.createElement("article");
      item.className = "activeFunction";

      const header = document.createElement("div");
      header.className = "activeFunctionHeader";
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = series.color;
      const name = document.createElement("strong");
      name.textContent = series.name;
      const value = document.createElement("span");
      value.className = "activationValue";
      value.textContent = series.currentValue.toFixed(3);
      header.append(swatch, name, value);

      if (this.state.functionSource === "decoded") {
        const decoded = series.functionNeurosynth || [];
        const summary = document.createElement("p");
        if (decoded.length === 0) {
          summary.textContent = "Neurosynth decoding is not available for this parcel.";
          item.append(header, summary);
        } else {
          summary.textContent = "Top Neurosynth meta-analytic terms (z-score):";
          const tags = document.createElement("div");
          tags.className = "systemTags";
          if (series.emotionTag) {
            const emo = document.createElement("span");
            emo.className = "emotionTag";
            emo.textContent = `${series.emotionTag.term} · ${series.emotionTag.z.toFixed(2)}`;
            tags.appendChild(emo);
          }
          for (const entry of decoded.slice(0, 10)) {
            const tag = document.createElement("span");
            tag.textContent = `${entry.term} · ${entry.z.toFixed(2)}`;
            tags.appendChild(tag);
          }
          item.append(header, summary, tags);
        }
      } else {
        const summary = document.createElement("p");
        summary.textContent = series.function?.summary || "Functional association is not available for this parcel.";
        const systems = document.createElement("div");
        systems.className = "systemTags";
        for (const system of series.function?.systems || []) {
          const tag = document.createElement("span");
          tag.textContent = system;
          systems.appendChild(tag);
        }
        const functions = document.createElement("ul");
        functions.className = "functionList";
        for (const fn of (series.function?.functions || []).slice(0, 3)) {
          const li = document.createElement("li");
          li.textContent = fn;
          functions.appendChild(li);
        }
        item.append(header, systems, summary, functions);
      }
      this.activeFunctions.appendChild(item);
    }
  }

  pause() {
    this.state.playing = false;
    this.state.syncingNativeControls = true;
    if (!this.sourceVideo.paused) this.sourceVideo.pause();
    if (!this.brainVideo.paused) this.brainVideo.pause();
    this.state.syncingNativeControls = false;
    this.playButton.textContent = "Play";
  }

  async play() {
    this.state.playing = true;
    this.playButton.textContent = "Pause";
    await Promise.allSettled([this.sourceVideo.play(), this.brainVideo.play()]);
  }

  tick() {
    if (this.state.playing && !this.state.dragging) {
      const t = this.sourceVideo.currentTime;
      if (this.sourceVideo.ended || this.brainVideo.ended || t >= this.state.duration - 0.02) {
        this.pause();
        this.seekTo(this.state.duration);
      } else {
        if (Math.abs(this.brainVideo.currentTime - t) > 0.08 && !this.brainVideo.ended) {
          this.brainVideo.currentTime = t;
        }
        this.renderAtTime(t);
      }
    }
  }

  _handleNativePlay() {
    if (!this.state.playing) this.play();
  }

  _handleNativePause(event) {
    if (this.state.syncingNativeControls || event.currentTarget.ended) return;
    if (this.state.playing) this.pause();
  }

  _handleSourceTimelineChange() {
    if (this.state.syncingNativeControls || this.state.dragging) return;
    const t = Math.min(this.sourceVideo.currentTime, this.state.duration);
    if (Math.abs(this.brainVideo.currentTime - t) > 0.08) {
      this.brainVideo.currentTime = t;
    }
    this.renderAtTime(t);
  }

  _handleBrainSeek() {
    if (this.state.syncingNativeControls || this.state.dragging) return;
    const t = Math.min(this.brainVideo.currentTime, this.state.duration);
    if (Math.abs(this.sourceVideo.currentTime - t) > 0.08) {
      this.sourceVideo.currentTime = t;
    }
    this.renderAtTime(t);
  }

  setBrainMode(mode) {
    if (mode === "3d") {
      const other = globalState.active3dPanel;
      if (other && other !== this) {
        other.setBrainMode("video");
      }
      globalState.active3dPanel = this;
    } else if (globalState.active3dPanel === this) {
      globalState.active3dPanel = null;
    }
    this.state.brainMode = mode;
    const is3d = mode === "3d";
    this.brainVideo.hidden = is3d;
    this.brain3dPane.hidden = !is3d;
    this.brainVideoTab.classList.toggle("active", !is3d);
    this.brain3dTab.classList.toggle("active", is3d);
    this.brainVideoTab.setAttribute("aria-selected", String(!is3d));
    this.brain3dTab.setAttribute("aria-selected", String(is3d));
    if (is3d) this.ensureInteractiveBrain();
  }

  async ensureInteractiveBrain() {
    if (!this.state.data) return;
    this.brain3dStatus.textContent = "Loading 3D surface...";
    if (!this.state.mesh) {
      this.state.mesh = await fetchJson(`/api/runs/${this.state.data.run.id}/brain-mesh`);
      const bg = this.state.mesh.bg;
      this.state.mesh.bgMin = Math.min(...bg);
      this.state.mesh.bgMax = Math.max(...bg);
    }
    await this.updateInteractiveBrain(true);
  }

  async _frameValues(second) {
    const key = `${this.state.data.run.id}:${second}`;
    if (!this.state.frameCache.has(key)) {
      const payload = await fetchJson(`/api/runs/${this.state.data.run.id}/brain-frame/${second}`);
      this.state.frameCache.set(key, payload);
    }
    return this.state.frameCache.get(key);
  }

  async updateInteractiveBrain(force = false) {
    if (this.state.brainMode !== "3d" || !this.state.mesh || !this.state.data) return;
    const second = Math.max(0, Math.min(this.state.data.seconds.length - 1, Math.round(Number(this.scrubber.value))));
    if (force || second !== this.state.brainFrameSecond) {
      this.state.currentFrame = await this._frameValues(second);
      this.state.brainFrameSecond = second;
    }
    this.drawBrain();
  }

  _resizeBrainCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const width = this.brainCanvas.clientWidth;
    const height = this.brainCanvas.clientHeight;
    if (this.brainCanvas.width !== Math.round(width * ratio) || this.brainCanvas.height !== Math.round(height * ratio)) {
      this.brainCanvas.width = Math.round(width * ratio);
      this.brainCanvas.height = Math.round(height * ratio);
    }
    this.brainCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { width, height };
  }

  _rotatePoint(point) {
    const mesh = this.state.mesh;
    const cx = mesh.center[0];
    const cy = mesh.center[1];
    const cz = mesh.center[2];
    const scale = mesh.scale || 1;
    const x = (point[0] - cx) / scale;
    const y = (point[1] - cy) / scale;
    const z = (point[2] - cz) / scale;
    return applyMatrix(this.state.rotationMatrix, x, y, z);
  }

  _valueColor(value, bgValue, limit) {
    const mesh = this.state.mesh;
    const amount = Math.min(Math.abs(value) / Math.max(limit, 1e-6), 1);
    const bgNorm = (bgValue - mesh.bgMin) / Math.max(mesh.bgMax - mesh.bgMin, 1e-6);
    const foldedSulci = 210 + bgNorm * 38;
    const base = [foldedSulci, foldedSulci, foldedSulci];

    if (amount < 0.18) {
      return `rgb(${base[0]}, ${base[1]}, ${base[2]})`;
    }
    const hot = Math.min((amount - 0.18) / 0.82, 1);
    const red = [245, 34, 25];
    const orange = [255, 120, 28];
    const yellow = [255, 244, 120];
    const target = hot < 0.62
      ? red.map((channel, index) => mix(channel, orange[index], hot / 0.62))
      : orange.map((channel, index) => mix(channel, yellow[index], (hot - 0.62) / 0.38));
    const blend = 0.42 + hot * 0.52;
    return `rgb(${mix(base[0], target[0], blend)}, ${mix(base[1], target[1], blend)}, ${mix(base[2], target[2], blend)})`;
  }

  drawBrain() {
    if (!this.state.mesh || !this.state.currentFrame) return;
    const ctx = this.brainCtx;
    const { width, height } = this._resizeBrainCanvas();
    ctx.clearRect(0, 0, width, height);
    const bgGradient = ctx.createRadialGradient(
      width * 0.5, height * 0.48, 20,
      width * 0.5, height * 0.48, Math.max(width, height) * 0.62,
    );
    bgGradient.addColorStop(0, "#101010");
    bgGradient.addColorStop(0.62, "#050505");
    bgGradient.addColorStop(1, "#000000");
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, width, height);

    const coords = this.state.mesh.coords;
    const projected = coords.map((point) => {
      const [x, y, z] = this._rotatePoint(point);
      const perspective = 1.95 / (2.65 - z);
      const s = Math.min(width, height) * 0.54 * this.state.zoom * perspective;
      return [width * 0.5 + x * s, height * 0.53 - y * s, z];
    });

    const values = this.state.currentFrame.values;
    const limit = this.state.currentFrame.limit;
    const faces = this.state.mesh.faces.map((face) => {
      const a = projected[face[0]];
      const b = projected[face[1]];
      const c = projected[face[2]];
      const depth = (a[2] + b[2] + c[2]) / 3;
      const value = (values[face[0]] + values[face[1]] + values[face[2]]) / 3;
      const bg = (this.state.mesh.bg[face[0]] + this.state.mesh.bg[face[1]] + this.state.mesh.bg[face[2]]) / 3;
      return { face, depth, value, bg };
    });
    faces.sort((a, b) => a.depth - b.depth);

    for (const item of faces) {
      const [i, j, k] = item.face;
      const a = projected[i];
      const b = projected[j];
      const c = projected[k];
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.lineTo(c[0], c[1]);
      ctx.closePath();
      ctx.fillStyle = this._valueColor(item.value, item.bg, limit);
      ctx.fill();
      ctx.strokeStyle = "rgba(30, 30, 30, 0.09)";
      ctx.lineWidth = 0.28;
      ctx.stroke();
    }

    ctx.fillStyle = "#b9c4ce";
    ctx.font = "12px system-ui";
    ctx.fillText(
      `3D cortical surface (${this.state.mesh.mesh}) | second ${this.state.brainFrameSecond} | drag a ring or empty space`,
      14,
      height - 16,
    );
    this._drawActivityLegend(width);
    this._drawRotationRings(width, height);
    this.brain3dStatus.textContent =
      `${this.state.mesh.mesh} preview. White cortex with red/yellow absolute activity; drag rings to rotate.`;
  }

  _drawActivityLegend(width) {
    const ctx = this.brainCtx;
    const x = width - 210;
    const y = 18;
    const w = 150;
    const h = 7;
    const gradient = ctx.createLinearGradient(x, y, x + w, y);
    gradient.addColorStop(0, "#1a1a1a");
    gradient.addColorStop(0.35, "#cc1010");
    gradient.addColorStop(0.7, "#ff7a1c");
    gradient.addColorStop(1, "#fff478");
    ctx.save();
    ctx.fillStyle = "#d9e2ec";
    ctx.font = "12px system-ui";
    ctx.fillText("Low", x, y - 6);
    ctx.fillText("High", x + w - 28, y - 6);
    ctx.fillStyle = gradient;
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "#d9e2ec";
    ctx.fillText("Activity", x + 49, y + 22);
    ctx.restore();
  }

  _projectControlPoint(point, width, height) {
    const [x, y, z] = applyMatrix(this.state.rotationMatrix, point[0], point[1], point[2]);
    const perspective = 1.95 / (2.65 - z);
    const s = Math.min(width, height) * 0.54 * this.state.zoom * perspective;
    return [width * 0.5 + x * s, height * 0.53 - y * s, z];
  }

  _drawRotationRings(width, height) {
    const ctx = this.brainCtx;
    const rings = [
      { axis: "x", label: "X", color: "#ff6b6b" },
      { axis: "y", label: "Y", color: "#4dd091" },
      { axis: "z", label: "Z", color: "#6aa7ff" },
    ];
    const radius = 1.58;
    const steps = 128;
    this.state.ringPaths = [];
    ctx.save();
    ctx.lineCap = "round";
    ctx.font = "700 13px system-ui";
    for (const ring of rings) {
      const points = [];
      for (let i = 0; i <= steps; i++) {
        const p = ring.axis === "x"
          ? [0, Math.cos((Math.PI * 2 * i) / steps) * radius, Math.sin((Math.PI * 2 * i) / steps) * radius]
          : ring.axis === "y"
          ? [Math.cos((Math.PI * 2 * i) / steps) * radius, 0, Math.sin((Math.PI * 2 * i) / steps) * radius]
          : [Math.cos((Math.PI * 2 * i) / steps) * radius, Math.sin((Math.PI * 2 * i) / steps) * radius, 0];
        points.push(this._projectControlPoint(p, width, height));
      }
      this.state.ringPaths.push({ axis: ring.axis, points });
      ctx.shadowColor = ring.color;
      ctx.shadowBlur = this.state.activeRing === ring.axis ? 18 : 8;
      ctx.globalAlpha = this.state.activeRing === ring.axis ? 0.98 : 0.54;
      ctx.lineWidth = this.state.activeRing === ring.axis ? 6 : 3;
      ctx.strokeStyle = ring.color;
      ctx.beginPath();
      points.forEach((point, index) => {
        if (index === 0) ctx.moveTo(point[0], point[1]);
        else ctx.lineTo(point[0], point[1]);
      });
      ctx.stroke();
      const labelPoint = points[Math.floor(steps * 0.11)];
      ctx.shadowBlur = 10;
      ctx.globalAlpha = 1;
      ctx.fillStyle = ring.color;
      ctx.beginPath();
      ctx.arc(labelPoint[0] + 11, labelPoint[1] - 11, 12, 0, Math.PI * 2);
      ctx.fillStyle = "#071018";
      ctx.fill();
      ctx.strokeStyle = ring.color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = ring.color;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.shadowBlur = 0;
      ctx.fillText(ring.label, labelPoint[0] + 6, labelPoint[1] - 6);
    }
    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";
    ctx.restore();
  }

  setBrainView(view) {
    const views = {
      top: { matrix: [[1,0,0],[0,1,0],[0,0,1]], zoom: 1.2 },
      left: { matrix: [[0,1,0],[0,0,1],[1,0,0]], zoom: 1.18 },
      right: { matrix: [[0,-1,0],[0,0,1],[-1,0,0]], zoom: 1.18 },
      front: { matrix: [[1,0,0],[0,0,1],[0,1,0]], zoom: 1.12 },
      reset: { matrix: eulerMatrix(0, 0), zoom: 1.2 },
    };
    const next = views[view];
    if (!next) return;
    this.state.rotationMatrix = next.matrix;
    this.state.zoom = next.zoom;
    this.drawBrain();
  }

  _canvasPoint(event) {
    const rect = this.brainCanvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  _segmentDistance(point, a, b) {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const lengthSq = dx * dx + dy * dy;
    if (lengthSq === 0) return Math.hypot(point.x - a[0], point.y - a[1]);
    const t = Math.max(0, Math.min(1, ((point.x - a[0]) * dx + (point.y - a[1]) * dy) / lengthSq));
    return Math.hypot(point.x - (a[0] + t * dx), point.y - (a[1] + t * dy));
  }

  _hitRotationRing(point) {
    let best = null;
    for (const path of this.state.ringPaths) {
      for (let i = 1; i < path.points.length; i++) {
        const distance = this._segmentDistance(point, path.points[i - 1], path.points[i]);
        if (!best || distance < best.distance) {
          best = { axis: path.axis, distance, path, index: i };
        }
      }
    }
    if (!best || best.distance >= 24) return null;
    const a = best.path.points[best.index - 1];
    const b = best.path.points[best.index];
    const tx = b[0] - a[0];
    const ty = b[1] - a[1];
    const len = Math.hypot(tx, ty) || 1;
    return { axis: best.axis, tangent: [tx / len, ty / len] };
  }

  _brainPointerDown(event) {
    const point = this._canvasPoint(event);
    this.state.brainDragging = true;
    this.state.brainPointer = { x: event.clientX, y: event.clientY };
    const hit = this._hitRotationRing(point);
    this.state.activeRing = hit ? hit.axis : null;
    this.state.activeRingTangent = hit ? hit.tangent : null;
    this.brainCanvas.setPointerCapture(event.pointerId);
    this.drawBrain();
  }

  _brainPointerMove(event) {
    if (!this.state.brainDragging || !this.state.brainPointer) return;
    const dx = event.clientX - this.state.brainPointer.x;
    const dy = event.clientY - this.state.brainPointer.y;
    this.state.brainPointer = { x: event.clientX, y: event.clientY };
    if (this.state.activeRing && this.state.activeRingTangent) {
      const t = this.state.activeRingTangent;
      const proj = dx * t[0] + dy * t[1];
      const radius = Math.min(this.brainCanvas.clientWidth, this.brainCanvas.clientHeight) * 0.4 || 200;
      const deltaAngle = Math.max(-0.12, Math.min(0.12, proj / radius));
      this.state.rotationMatrix = multiplyMatrix(
        this.state.rotationMatrix,
        rotationMatrixForAxis(this.state.activeRing, deltaAngle),
      );
    } else {
      const delta = multiplyMatrix(rotationYMatrix(-dx * 0.008), rotationXMatrix(dy * 0.008));
      this.state.rotationMatrix = multiplyMatrix(delta, this.state.rotationMatrix);
    }
    this.drawBrain();
  }

  _brainPointerUp() {
    this.state.brainDragging = false;
    this.state.brainPointer = null;
    this.state.activeRing = null;
    this.state.activeRingTangent = null;
    this.drawBrain();
  }
}

async function loadRunsList() {
  const payload = await fetchJson("/api/runs");
  runsCache.runs = payload.runs;
  runsCache.loaded = true;
  for (const panel of panels) {
    const current = panel.runSelect.value;
    panel._populateRunSelect(current);
  }
  if (!runsCache.runs.length) {
    globalMeta.textContent = "No completed TRIBE runs found in outputs. Use + New video below to create one.";
  } else {
    globalMeta.textContent = `${runsCache.runs.length} run(s) available.`;
  }
  populateExistingRunSelect();
}

async function addPanel(runId) {
  const panel = new ReviewPanel();
  panels.push(panel);
  if (runId) await panel.loadRun(runId);
  persistOpenPanels();
  return panel;
}

const OPEN_PANELS_KEY = "tribeReview.openPanels.v1";
function persistOpenPanels() {
  try {
    const ids = panels.map((p) => p.runSelect.value).filter(Boolean);
    localStorage.setItem(OPEN_PANELS_KEY, JSON.stringify(ids));
  } catch (_) {}
}
function loadPersistedPanelIds() {
  try {
    const raw = localStorage.getItem(OPEN_PANELS_KEY);
    if (!raw) return [];
    const ids = JSON.parse(raw);
    return Array.isArray(ids) ? ids : [];
  } catch (_) {
    return [];
  }
}

function tickAll() {
  for (const panel of panels) panel.tick();
  requestAnimationFrame(tickAll);
}

window.addEventListener("resize", () => {
  for (const panel of panels) {
    if (panel.state.data) panel.drawChart();
    if (panel.state.brainMode === "3d") panel.drawBrain();
  }
});

const newVideoButton = document.querySelector("#newVideoButton");
const uploadPanel = document.querySelector("#uploadPanel");
const closeUploadButton = document.querySelector("#closeUploadButton");
const existingRunSelect = document.querySelector("#existingRunSelect");
const addExistingRunButton = document.querySelector("#addExistingRunButton");
const uploadForm = document.querySelector("#uploadForm");
const uploadFile = document.querySelector("#uploadFile");
const uploadBackend = document.querySelector("#uploadBackend");
const uploadFlavor = document.querySelector("#uploadFlavor");
const uploadMaxSeconds = document.querySelector("#uploadMaxSeconds");
const uploadWithText = document.querySelector("#uploadWithText");
const uploadSubmit = document.querySelector("#uploadSubmit");
const uploadStatus = document.querySelector("#uploadStatus");

function populateExistingRunSelect() {
  if (!existingRunSelect) return;
  const current = existingRunSelect.value;
  existingRunSelect.innerHTML = "";
  for (const run of runsCache.runs) {
    const option = document.createElement("option");
    option.value = run.id;
    option.textContent = `${run.name} (${run.seconds}s)`;
    existingRunSelect.appendChild(option);
  }
  if (runsCache.runs.some((run) => run.id === current)) {
    existingRunSelect.value = current;
  }
  const hasRuns = runsCache.runs.length > 0;
  existingRunSelect.disabled = !hasRuns;
  addExistingRunButton.disabled = !hasRuns;
}

function setUploadOpen(open) {
  uploadPanel.hidden = !open;
  if (open) {
    if (!existingRunSelect.disabled) existingRunSelect.focus();
    else uploadFile.focus();
    uploadPanel.scrollIntoView({ behavior: "smooth", block: "end" });
  }
}

newVideoButton.addEventListener("click", () => setUploadOpen(uploadPanel.hidden));
closeUploadButton.addEventListener("click", () => setUploadOpen(false));

addExistingRunButton.addEventListener("click", async () => {
  if (!existingRunSelect.value) return;
  addExistingRunButton.disabled = true;
  uploadStatus.textContent = "Adding existing run...";
  try {
    await addPanel(existingRunSelect.value);
    setUploadOpen(false);
    uploadStatus.textContent = "";
  } catch (error) {
    uploadStatus.textContent = `Error: ${error.message}`;
  } finally {
    addExistingRunButton.disabled = runsCache.runs.length === 0;
  }
});

uploadBackend.addEventListener("change", () => {
  uploadFlavor.disabled = uploadBackend.value !== "hf";
});

async function pollJob(jobId) {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 2500));
    const job = await fetchJson(`/api/jobs/${jobId}`);
    const stage = job.stage || job.status || "...";
    uploadStatus.textContent = `${stage}: ${job.message || ""}`;
    if (job.status === "done") return job;
    if (job.status === "error") throw new Error(job.message || "Job failed");
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!uploadFile.files.length) return;
  const formData = new FormData();
  formData.append("file", uploadFile.files[0]);
  formData.append("backend", uploadBackend.value);
  formData.append("with_text", uploadWithText.checked ? "true" : "false");
  formData.append("max_timesteps", String(parseInt(uploadMaxSeconds.value || "0", 10)));
  formData.append("flavor", uploadFlavor.value);
  uploadSubmit.disabled = true;
  uploadStatus.textContent = "Uploading video...";
  try {
    const response = await fetch("/api/upload", { method: "POST", body: formData });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`${response.status} ${text}`);
    }
    const { job_id } = await response.json();
    uploadStatus.textContent = "Job queued. This can take several minutes...";
    const job = await pollJob(job_id);
    uploadStatus.textContent = "Run complete. Loading...";
    await loadRunsList();
    if (job.run_id) {
      await addPanel(job.run_id);
    }
    setUploadOpen(false);
    uploadForm.reset();
    uploadStatus.textContent = "";
  } catch (error) {
    uploadStatus.textContent = `Error: ${error.message}`;
  } finally {
    uploadSubmit.disabled = false;
  }
});

(async function init() {
  try {
    await loadRunsList();
    const available = new Set(runsCache.runs.map((r) => r.id));
    const saved = loadPersistedPanelIds().filter((id) => available.has(id));
    if (saved.length) {
      for (const id of saved) await addPanel(id);
    } else if (runsCache.runs.length) {
      await addPanel(runsCache.runs[0].id);
    }
  } catch (error) {
    globalMeta.textContent = error.message;
  }
  tickAll();
})();
