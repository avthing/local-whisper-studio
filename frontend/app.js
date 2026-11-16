const state = {
  jobs: [],
  currentJobId: null,
  currentJob: null,
  peaks: [],
  filterSpeaker: "ALL",
  pollTimer: null,
  searchQuery: "",
  recorder: null,
  recStreams: [],
  recChunks: [],
  recInterval: null,
  recSeconds: 0,
  catalog: { asr_models: [], diarization_modes: [] },
  pipelineMode: "default", // 'default' | 'rerun'
  pipelineCfg: {
    asr_model: "breeze-asr-25",
    diarization_mode: "sherpa-strict",
    num_speakers: -1,
    enhance_audio: true,
    prompt: "",
  },
};

const $ = (id) => document.getElementById(id);

// --- Theme Init ---
const savedTheme = localStorage.getItem("lws_theme") || "light";
document.documentElement.setAttribute("data-theme", savedTheme);
$("btn-theme").textContent = savedTheme === "dark" ? "☀️" : "🌙";
$("btn-theme").addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("lws_theme", next);
  $("btn-theme").textContent = next === "dark" ? "☀️" : "🌙";
  drawWaveform();
});

function fmtTs(sec) {
  const t = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = t % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${pad(h)}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

async function api(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const data = await res.json();
      msg = data.detail || msg;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

// --- Jobs List ---
async function loadJobs() {
  const data = await api("/api/jobs");
  state.jobs = data.jobs || [];
  renderJobList();
  if (!state.currentJobId && state.jobs.length > 0) {
    selectJob(state.jobs[0].job_id);
  }
}

function renderJobList() {
  const container = $("job-list");
  container.innerHTML = "";
  for (const job of state.jobs) {
    const el = document.createElement("div");
    el.className = `job-item ${job.job_id === state.currentJobId ? "active" : ""}`;
    el.innerHTML = `
      <div class="job-item-top">
        <span class="job-item-title" title="${escapeHtml(job.title)}">${escapeHtml(job.title)}</span>
        <button class="job-item-del" title="刪除">✕</button>
      </div>
      <div class="job-item-sub">
        <span>${job.status === "completed" ? fmtTs(job.duration) : escapeHtml(job.stage || job.status)}</span>
        <span>${job.progress ?? 0}%</span>
      </div>
    `;
    el.addEventListener("click", (e) => {
      if (e.target.classList.contains("job-item-del")) {
        e.stopPropagation();
        deleteJob(job.job_id);
      } else {
        selectJob(job.job_id);
      }
    });
    container.appendChild(el);
  }
}

async function deleteJob(jobId) {
  if (!confirm("確定刪除這筆紀錄與音檔嗎？")) return;
  await api(`/api/jobs/${jobId}`, { method: "DELETE" });
  if (state.currentJobId === jobId) {
    state.currentJobId = null;
    state.currentJob = null;
    $("workspace-body").style.display = "none";
    $("waveform-bar").style.display = "none";
    $("topbar-actions").style.display = "none";
    $("empty-state").style.display = "flex";
  }
  await loadJobs();
}

async function selectJob(jobId) {
  state.currentJobId = jobId;
  state.filterSpeaker = "ALL";
  state.peaks = [];
  renderJobList();
  await refreshCurrentJob();
}

async function refreshCurrentJob() {
  if (!state.currentJobId) return;
  const job = await api(`/api/jobs/${state.currentJobId}`);
  state.currentJob = job;

  $("empty-state").style.display = "none";
  $("job-title").textContent = job.title;
  $("job-meta").textContent = `${job.original_filename} ・ ${fmtTs(job.duration)} ・ ${Object.keys(job.speakers || {}).length} 位語者`;
  $("topbar-actions").style.display = "flex";

  if (job.status === "queued" || job.status === "processing") {
    $("progress-banner").style.display = "block";
    const pct = job.progress || 0;
    $("progress-stage").textContent = job.stage || "處理中...";
    $("progress-pct").textContent = `${pct}%`;
    $("progress-fill").style.width = `${pct}%`;

    const el = job.elapsed_sec ? `已耗時 ${fmtTs(job.elapsed_sec)}` : "";
    const rem = job.eta_remaining_sec ? `預估剩餘約 ${fmtTs(job.eta_remaining_sec)}` : "";
    $("progress-eta").textContent = [el, rem].filter(Boolean).join(" ・ ");

    document.querySelectorAll("#progress-steps span").forEach((sp) => {
      const stepNum = parseInt(sp.dataset.step, 10);
      sp.classList.remove("done", "active");
      const curStep = pct < 10 ? 1 : pct < 60 ? 2 : pct < 95 ? 3 : 4;
      if (stepNum < curStep) sp.classList.add("done");
      else if (stepNum === curStep) sp.classList.add("active");
    });
    schedulePoll();
  } else if (job.status === "failed") {
    $("progress-banner").style.display = "block";
    $("progress-stage").textContent = `❌ 處理失敗：${job.error || "未知錯誤"}`;
    $("progress-eta").textContent = "";
    $("progress-pct").textContent = "";
    $("progress-fill").style.width = "100%";
  } else {
    $("progress-banner").style.display = "none";
  }

  if (job.status === "completed") {
    $("workspace-body").style.display = "flex";
    $("waveform-bar").style.display = "flex";
    const audio = $("main-audio");
    const targetSrc = `/api/jobs/${job.job_id}/audio`;
    if (!audio.src.endsWith(targetSrc)) {
      audio.src = targetSrc;
      $("btn-play-pause").textContent = "▶";
    }
    $("summary-output").value = job.summary || "";
    renderSpeakers(job);
    renderFilterPills(job);
    renderTranscript(job);
    renderSpeakerRibbon(job);
    if (state.peaks.length === 0) {
      const wf = await api(`/api/jobs/${job.job_id}/waveform`);
      state.peaks = wf.peaks || [];
      drawWaveform();
    } else {
      drawWaveform();
    }
  } else {
    $("workspace-body").style.display = "none";
    $("waveform-bar").style.display = "none";
  }
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(async () => {
    await loadJobs();
    if (state.currentJobId) await refreshCurrentJob();
  }, 2000);
}

// --- Waveform & Speaker Ribbon ---
function drawWaveform() {
  const canvas = $("waveform-canvas");
  if (!canvas || !state.peaks.length) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0) return;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const n = state.peaks.length;
  const barGap = 1.5;
  const barW = Math.max(1.5, (w - (n - 1) * barGap) / n);
  const audio = $("main-audio");
  const dur = audio.duration || state.currentJob?.duration || 1;
  const progressRatio = Math.min(1, Math.max(0, (audio.currentTime || 0) / dur));

  const styles = getComputedStyle(document.documentElement);
  const playedColor = styles.getPropertyValue("--accent").trim();
  const unplayedColor = styles.getPropertyValue("--wave-bg").trim();

  ctx.clearRect(0, 0, w, h);
  for (let i = 0; i < n; i++) {
    const amp = Math.max(0.08, state.peaks[i]);
    const barH = amp * (h - 6);
    const x = i * (barW + barGap);
    const y = (h - barH) / 2;
    ctx.fillStyle = i / n <= progressRatio ? playedColor : unplayedColor;
    ctx.beginPath();
    ctx.roundRect(x, y, barW, barH, 2);
    ctx.fill();
  }
}

$("waveform-canvas").addEventListener("click", (e) => {
  const rect = e.target.getBoundingClientRect();
  const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  const audio = $("main-audio");
  const dur = audio.duration || state.currentJob?.duration || 0;
  if (dur > 0) {
    audio.currentTime = ratio * dur;
    audio.play();
  }
});

window.addEventListener("resize", () => drawWaveform());

function renderSpeakerRibbon(job) {
  const ribbon = $("speaker-ribbon");
  ribbon.innerHTML = "";
  const dur = job.duration || 1;
  const speakers = job.speakers || {};
  for (const seg of job.segments || []) {
    const leftPct = (seg.start / dur) * 100;
    const widthPct = Math.max(0.2, ((seg.end - seg.start) / dur) * 100);
    const div = document.createElement("div");
    div.className = "ribbon-seg";
    div.style.left = `${leftPct}%`;
    div.style.width = `${widthPct}%`;
    div.style.background = speakers[seg.speaker]?.color || "#8b95a7";
    ribbon.appendChild(div);
  }
}

// --- Speakers Panel + Merge ---
function renderSpeakers(job) {
  const list = $("speaker-list");
  list.innerHTML = "";
  const speakers = job.speakers || {};
  const entries = Object.entries(speakers);

  for (const [spkId, info] of entries) {
    const initial = (info.display_name || spkId).slice(0, 2).toUpperCase();
    const otherOpts = entries
      .filter(([otherId]) => otherId !== spkId)
      .map(([otherId, otherInfo]) => `<option value="${otherId}">${escapeHtml(otherInfo.display_name)}</option>`)
      .join("");

    const card = document.createElement("div");
    card.className = "speaker-card";
    card.innerHTML = `
      <div class="spk-top">
        <span class="spk-avatar" style="background:${info.color}">${escapeHtml(initial)}</span>
        <input class="spk-input" data-spk="${spkId}" value="${escapeHtml(info.display_name)}" />
        ${info.sample_file ? `<button class="spk-sample-btn" data-sample="${info.sample_file}" title="試聽 6 秒原音">🔊</button>` : ""}
      </div>
      <div class="spk-stats">
        <span>${fmtTs(info.duration_sec)} (${info.ratio_pct}%)</span>
        <span>${info.segment_count} 段</span>
      </div>
      <div class="spk-bar-bg">
        <div class="spk-bar-fill" style="width:${info.ratio_pct}%;background:${info.color}"></div>
      </div>
      ${entries.length > 1 ? `
      <div class="spk-merge-row">
        <select class="spk-merge-select">
          <option value="">合併此語者至...</option>
          ${otherOpts}
        </select>
      </div>` : ""}
    `;

    const sampleBtn = card.querySelector(".spk-sample-btn");
    if (sampleBtn) {
      sampleBtn.addEventListener("click", () => {
        const sAudio = $("sample-audio");
        $("main-audio").pause();
        sAudio.src = `/api/jobs/${job.job_id}/samples/${sampleBtn.dataset.sample}`;
        sAudio.play();
      });
    }

    const mergeSel = card.querySelector(".spk-merge-select");
    if (mergeSel) {
      mergeSel.addEventListener("change", async () => {
        const targetId = mergeSel.value;
        if (!targetId) return;
        const targetName = speakers[targetId]?.display_name || targetId;
        if (!confirm(`確定將「${info.display_name}」的所有片段合併至「${targetName}」嗎？`)) {
          mergeSel.value = "";
          return;
        }
        await api(`/api/jobs/${job.job_id}/speakers/merge`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source_id: spkId, target_id: targetId }),
        });
        await refreshCurrentJob();
      });
    }

    list.appendChild(card);
  }
}

$("btn-save-speakers").addEventListener("click", async () => {
  if (!state.currentJobId) return;
  const mapping = {};
  document.querySelectorAll(".spk-input").forEach((inp) => {
    mapping[inp.dataset.spk] = inp.value;
  });
  await api(`/api/jobs/${state.currentJobId}/speakers`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mapping }),
  });
  await refreshCurrentJob();
});

// --- Filter Pills ---
function renderFilterPills(job) {
  const bar = $("filter-pills");
  bar.innerHTML = "";
  const speakers = job.speakers || {};
  const total = (job.segments || []).length;

  const addPill = (id, label, color = null) => {
    const btn = document.createElement("button");
    btn.className = `filter-pill ${state.filterSpeaker === id ? "active" : ""}`;
    btn.innerHTML = `${color ? `<span class="pill-dot" style="background:${color}"></span>` : ""}${escapeHtml(label)}`;
    btn.addEventListener("click", () => {
      state.filterSpeaker = id;
      renderFilterPills(job);
      renderTranscript(job);
    });
    bar.appendChild(btn);
  };

  addPill("ALL", `全部 (${total})`);
  for (const [spkId, info] of Object.entries(speakers)) {
    addPill(spkId, `${info.display_name} (${info.segment_count})`, info.color);
  }
}

// --- Transcript Cards & Inline Reassign ---
function renderTranscript(job) {
  const list = $("transcript-list");
  list.innerHTML = "";
  const speakers = job.speakers || {};
  const q = state.searchQuery.trim().toLowerCase();

  (job.segments || []).forEach((seg, idx) => {
    if (state.filterSpeaker !== "ALL" && seg.speaker !== state.filterSpeaker) return;
    const spkInfo = speakers[seg.speaker] || { display_name: seg.speaker, color: "#8b95a7" };
    if (q && !seg.text.toLowerCase().includes(q) && !spkInfo.display_name.toLowerCase().includes(q)) return;

    const initial = (spkInfo.display_name || seg.speaker).slice(0, 1).toUpperCase();
    const card = document.createElement("div");
    card.className = "seg-card";
    card.dataset.index = idx;
    card.dataset.start = seg.start;
    card.dataset.end = seg.end;
    card.innerHTML = `
      <div class="seg-avatar" style="background:${spkInfo.color}">${escapeHtml(initial)}</div>
      <div class="seg-body">
        <div class="seg-meta">
          <div class="seg-meta-left">
            <button class="seg-spk-btn" style="color:${spkInfo.color}" title="點擊可切換此段語者">${escapeHtml(spkInfo.display_name)} ▾</button>
            <span class="seg-time">${fmtTs(seg.start)}</span>
          </div>
          <div class="seg-actions">
            <button class="mini-act act-edit" title="編輯文字">✎</button>
            <button class="mini-act act-copy" title="複製這段">📋</button>
          </div>
        </div>
        <div class="seg-text">${escapeHtml(seg.text)}</div>
      </div>
    `;

    const textEl = card.querySelector(".seg-text");
    const spkBtn = card.querySelector(".seg-spk-btn");

    card.addEventListener("click", (e) => {
      if (textEl.getAttribute("contenteditable") === "true") return;
      if (e.target.closest(".seg-spk-btn") || e.target.closest(".mini-act")) return;
      const audio = $("main-audio");
      $("sample-audio").pause();
      audio.currentTime = seg.start;
      audio.play();
    });

    spkBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      openSpeakerPopover(spkBtn, job, idx);
    });

    card.querySelector(".act-copy").addEventListener("click", (e) => {
      e.stopPropagation();
      navigator.clipboard.writeText(`[${fmtTs(seg.start)}] ${spkInfo.display_name}：${seg.text}`);
    });

    const startEdit = () => {
      textEl.setAttribute("contenteditable", "true");
      textEl.focus();
    };
    card.querySelector(".act-edit").addEventListener("click", (e) => {
      e.stopPropagation();
      startEdit();
    });
    textEl.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      startEdit();
    });
    textEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        textEl.blur();
      }
    });
    textEl.addEventListener("blur", async () => {
      if (textEl.getAttribute("contenteditable") !== "true") return;
      textEl.removeAttribute("contenteditable");
      const newText = textEl.textContent.trim();
      if (newText !== seg.text) {
        await api(`/api/jobs/${job.job_id}/segments/${idx}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: newText }),
        });
        seg.text = newText;
      }
    });

    list.appendChild(card);
  });
}

function openSpeakerPopover(anchorEl, job, segIndex) {
  const pop = $("spk-popover");
  const rect = anchorEl.getBoundingClientRect();
  pop.innerHTML = "";
  for (const [spkId, info] of Object.entries(job.speakers || {})) {
    const item = document.createElement("button");
    item.className = "spk-pop-item";
    item.innerHTML = `<span class="pill-dot" style="background:${info.color}"></span>${escapeHtml(info.display_name)}`;
    item.addEventListener("click", async () => {
      pop.style.display = "none";
      await api(`/api/jobs/${job.job_id}/segments/${segIndex}/speaker`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speaker_id: spkId }),
      });
      await refreshCurrentJob();
    });
    pop.appendChild(item);
  }
  pop.style.left = `${rect.left}px`;
  pop.style.top = `${rect.bottom + 4}px`;
  pop.style.display = "flex";
}

document.addEventListener("click", (e) => {
  if (!e.target.closest("#spk-popover") && !e.target.closest(".seg-spk-btn")) {
    $("spk-popover").style.display = "none";
  }
});

$("search-input").addEventListener("input", (e) => {
  state.searchQuery = e.target.value;
  if (state.currentJob) renderTranscript(state.currentJob);
});

// --- Audio Player Sync ---
const mainAudio = $("main-audio");
$("btn-play-pause").addEventListener("click", () => {
  if (mainAudio.paused) mainAudio.play();
  else mainAudio.pause();
});
$("btn-skip-back").addEventListener("click", () => {
  mainAudio.currentTime = Math.max(0, mainAudio.currentTime - 5);
});
$("btn-skip-fwd").addEventListener("click", () => {
  mainAudio.currentTime = Math.min(mainAudio.duration || 0, mainAudio.currentTime + 5);
});
mainAudio.addEventListener("play", () => ($("btn-play-pause").textContent = "⏸"));
mainAudio.addEventListener("pause", () => ($("btn-play-pause").textContent = "▶"));
mainAudio.addEventListener("loadedmetadata", () => {
  $("player-duration").textContent = fmtTs(mainAudio.duration);
  drawWaveform();
});
mainAudio.addEventListener("timeupdate", () => {
  const t = mainAudio.currentTime;
  $("player-current").textContent = fmtTs(t);
  drawWaveform();
  highlightSegmentAt(t);
});
$("player-speed").addEventListener("change", (e) => {
  mainAudio.playbackRate = parseFloat(e.target.value);
});

function highlightSegmentAt(t) {
  let activeRow = null;
  document.querySelectorAll(".seg-card").forEach((row) => {
    const start = parseFloat(row.dataset.start);
    const end = parseFloat(row.dataset.end);
    const match = t >= start && t <= end + 0.3;
    row.classList.toggle("active", match);
    if (match) activeRow = row;
  });
  if (activeRow && $("chk-autoscroll").checked) {
    activeRow.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

// --- Upload & Dropzone ---
const dropzone = $("dropzone");
const fileInput = $("file-input");
dropzone.addEventListener("click", (e) => {
  if (e.target.closest("#btn-open-pipeline-cfg")) return;
  fileInput.click();
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) uploadFile(fileInput.files[0]);
});
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});

async function uploadFile(file, customTitle = "") {
  const form = new FormData();
  form.append("file", file);
  form.append("title", customTitle);
  form.append("num_speakers", String(state.pipelineCfg.num_speakers));
  form.append("asr_model", state.pipelineCfg.asr_model);
  form.append("diarization_mode", state.pipelineCfg.diarization_mode);
  form.append("enhance_audio", String(state.pipelineCfg.enhance_audio));
  form.append("prompt", state.pipelineCfg.prompt || "");
  const job = await api("/api/jobs/upload", { method: "POST", body: form });
  await loadJobs();
  await selectJob(job.job_id);
}

// --- Model & Diarization Catalog + Selector Modal ---
async function loadCatalogAndDefaults() {
  const [cat, cfg] = await Promise.all([api("/api/models"), api("/api/settings")]);
  state.catalog = cat;
  state.pipelineCfg.asr_model = cat.defaults.asr_model;
  state.pipelineCfg.diarization_mode = cat.defaults.diarization_mode;
  state.pipelineCfg.prompt = cfg.default_prompt || "";
  if (cfg.ai_provider) $("summary-provider").value = cfg.ai_provider;

  const asrSel = $("cfg-asr-model");
  asrSel.innerHTML = cat.asr_models
    .map((m) => `<option value="${m.id}">${escapeHtml(m.name)}</option>`)
    .join("");
  const diarSel = $("cfg-diar-mode");
  diarSel.innerHTML = cat.diarization_modes
    .map((d) => `<option value="${d.id}">${escapeHtml(d.name)}</option>`)
    .join("");

  updatePresetBadge();
}

function updatePresetBadge() {
  const asr = state.catalog.asr_models.find((m) => m.id === state.pipelineCfg.asr_model);
  const diar = state.catalog.diarization_modes.find((d) => d.id === state.pipelineCfg.diarization_mode);
  const asrLabel = asr ? asr.id : state.pipelineCfg.asr_model;
  const diarLabel = diar ? diar.name.split("（")[0] : state.pipelineCfg.diarization_mode;
  $("preset-summary").textContent = `${asrLabel} ・ ${diarLabel}`;
}

function renderAsrInfo() {
  const m = state.catalog.asr_models.find((x) => x.id === $("cfg-asr-model").value);
  if (!m) return;
  $("asr-info-box").innerHTML = `
    <div class="info-tag-row">
      <span class="info-badge">速度：${escapeHtml(m.speed)}</span>
      <span class="info-badge">準度：${escapeHtml(m.accuracy)}</span>
    </div>
    <div class="info-pros">✅ <strong>優點</strong>：${escapeHtml(m.pros)}</div>
    <div class="info-cons">⚠️ <strong>缺點</strong>：${escapeHtml(m.cons)}</div>
    <div>🎯 <strong>適合</strong>：${escapeHtml(m.recommended_for)}</div>
  `;
}

function renderDiarInfo() {
  const d = state.catalog.diarization_modes.find((x) => x.id === $("cfg-diar-mode").value);
  if (!d) return;
  const tokenTag = d.requires_token ? "需 HF Token" : "免 Token 全離線";
  $("diar-info-box").innerHTML = `
    <div class="info-tag-row">
      <span class="info-badge">${tokenTag}</span>
      <span class="info-badge">速度：${escapeHtml(d.speed)}</span>
    </div>
    <div class="info-pros">✅ <strong>優點</strong>：${escapeHtml(d.pros)}</div>
    <div class="info-cons">⚠️ <strong>缺點</strong>：${escapeHtml(d.cons)}</div>
    <div>🎯 <strong>適合</strong>：${escapeHtml(d.recommended_for)}</div>
  `;
}

function openPipelineModal(mode) {
  state.pipelineMode = mode;
  const job = mode === "rerun" ? state.currentJob : null;
  $("pipeline-modal-title").textContent =
    mode === "rerun" ? "🔄 換模型／辨別方式重新處理此錄音" : "🎛️ 預設語音模型與語者辨別方式";
  $("btn-apply-pipeline").textContent = mode === "rerun" ? "立即用此設定重跑" : "套用為預設上傳設定";

  $("cfg-asr-model").value = job?.asr_model || state.pipelineCfg.asr_model;
  $("cfg-diar-mode").value = job?.diarization_mode || state.pipelineCfg.diarization_mode;
  $("cfg-num-speakers").value = String(job?.num_speakers_hint ?? state.pipelineCfg.num_speakers);
  $("cfg-enhance-audio").checked = job ? job.enhance_audio !== false : state.pipelineCfg.enhance_audio;
  $("cfg-prompt").value = job?.prompt || state.pipelineCfg.prompt;

  if (job?.duration) {
    const durSel = $("wiz-duration");
    const customOpt = durSel.querySelector('option[data-custom="1"]');
    if (customOpt) customOpt.remove();
    const opt = document.createElement("option");
    opt.value = String(Math.round(job.duration));
    opt.dataset.custom = "1";
    opt.textContent = `此音檔真實長度 (${fmtTs(job.duration)})`;
    durSel.prepend(opt);
    durSel.value = opt.value;
  }

  renderAsrInfo();
  renderDiarInfo();
  updateWizardRecommendation();
  updateManualEta();
  $("pipeline-modal").style.display = "flex";
}

async function updateWizardRecommendation() {
  const res = await api("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      language: $("wiz-lang").value,
      scenario: $("wiz-scenario").value,
      priority: $("wiz-priority").value,
      duration_sec: parseFloat($("wiz-duration").value) || 3600,
    }),
  });
  const asrName = state.catalog.asr_models.find((m) => m.id === res.asr_model)?.name || res.asr_model;
  const diarName = state.catalog.diarization_modes.find((d) => d.id === res.diarization_mode)?.name || res.diarization_mode;
  const est = res.estimate;
  $("wiz-result-card").innerHTML = `
    <div class="rec-card-title">💡 為你推薦最佳組合</div>
    <div><strong>語音模型</strong>：${escapeHtml(asrName)}</div>
    <div><strong>辨別方式</strong>：${escapeHtml(diarName)}（${res.num_speakers === -1 ? "自動偵測人數" : `鎖定 ${res.num_speakers} 人`}・${res.enhance_audio ? "開啟遠場降噪均衡" : "原始音質"}）</div>
    <div><strong>推薦理由</strong>：${escapeHtml(res.reason)}</div>
    <div>⏱️ <strong>預估總耗時</strong>：約 <strong>${est.min_minutes} – ${est.max_minutes} 分鐘</strong>（聲紋約 ${Math.max(1, Math.round(est.diarization_sec / 60))} 分 ＋ 轉錄約 ${Math.max(1, Math.round(est.asr_sec / 60))} 分）</div>
  `;
  // Sync recommended picks into manual controls so "確認套用" uses them directly
  $("cfg-asr-model").value = res.asr_model;
  $("cfg-diar-mode").value = res.diarization_mode;
  $("cfg-num-speakers").value = String(res.num_speakers);
  $("cfg-enhance-audio").checked = res.enhance_audio;
  renderAsrInfo();
  renderDiarInfo();
  updateManualEta();
}

async function updateManualEta() {
  const dur = parseFloat($("wiz-duration").value) || 3600;
  const est = await api("/api/estimate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      duration_sec: dur,
      asr_model: $("cfg-asr-model").value,
      diarization_mode: $("cfg-diar-mode").value,
    }),
  });
  $("manual-eta-box").innerHTML = `⏱️ 以 ${fmtTs(dur)} 音檔估算，總處理時間約 <strong>${est.min_minutes} – ${est.max_minutes} 分鐘</strong>（聲紋約 ${Math.max(1, Math.round(est.diarization_sec / 60))} 分 ＋ 轉錄約 ${Math.max(1, Math.round(est.asr_sec / 60))} 分）`;
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const isWiz = btn.dataset.tab === "wizard";
    $("tab-wizard").style.display = isWiz ? "flex" : "none";
    $("tab-manual").style.display = isWiz ? "none" : "flex";
    if (!isWiz) updateManualEta();
  });
});

["wiz-lang", "wiz-scenario", "wiz-priority", "wiz-duration"].forEach((id) => {
  $(id).addEventListener("change", updateWizardRecommendation);
});

$("btn-open-pipeline-cfg").addEventListener("click", (e) => {
  e.stopPropagation();
  openPipelineModal("default");
});
$("btn-rerun-job").addEventListener("click", () => openPipelineModal("rerun"));
$("btn-close-pipeline").addEventListener("click", () => ($("pipeline-modal").style.display = "none"));
$("cfg-asr-model").addEventListener("change", () => {
  renderAsrInfo();
  updateManualEta();
});
$("cfg-diar-mode").addEventListener("change", () => {
  renderDiarInfo();
  updateManualEta();
});

$("btn-apply-pipeline").addEventListener("click", async () => {
  const payload = {
    asr_model: $("cfg-asr-model").value,
    diarization_mode: $("cfg-diar-mode").value,
    num_speakers: parseInt($("cfg-num-speakers").value, 10),
    enhance_audio: $("cfg-enhance-audio").checked,
    prompt: $("cfg-prompt").value,
  };
  $("pipeline-modal").style.display = "none";

  if (state.pipelineMode === "rerun" && state.currentJobId) {
    await api(`/api/jobs/${state.currentJobId}/rerun`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.peaks = [];
    await loadJobs();
    await refreshCurrentJob();
  } else {
    state.pipelineCfg = payload;
    updatePresetBadge();
  }
});

// --- Browser Live Recorder ---
$("btn-rec-start").addEventListener("click", async () => {
  const mode = $("rec-mode").value;
  try {
    const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.recStreams = [micStream];
    const ctx = new AudioContext();
    const dest = ctx.createMediaStreamDestination();
    ctx.createMediaStreamSource(micStream).connect(dest);

    if (mode === "mix") {
      const sysStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      state.recStreams.push(sysStream);
      const audioTracks = sysStream.getAudioTracks();
      if (audioTracks.length > 0) {
        ctx.createMediaStreamSource(new MediaStream(audioTracks)).connect(dest);
      } else {
        alert("未偵測到系統分頁音訊，將僅錄製麥克風（請記得勾選分享分頁音訊）。");
      }
    }

    const analyser = ctx.createAnalyser();
    ctx.createMediaStreamSource(dest.stream).connect(analyser);
    const dataArr = new Uint8Array(analyser.frequencyBinCount);

    const recorder = new MediaRecorder(dest.stream, { mimeType: "audio/webm" });
    state.recChunks = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) state.recChunks.push(e.data);
    };
    recorder.onstop = async () => {
      const blob = new Blob(state.recChunks, { type: "audio/webm" });
      const tsName = `即時錄音_${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}`;
      const file = new File([blob], `${tsName}.webm`, { type: "audio/webm" });
      state.recStreams.forEach((s) => s.getTracks().forEach((t) => t.stop()));
      ctx.close();
      await uploadFile(file, tsName);
    };

    recorder.start(1000);
    state.recorder = recorder;
    const recStartTime = Date.now();
    $("btn-rec-start").disabled = true;
    $("btn-rec-stop").disabled = false;

    state.recInterval = setInterval(() => {
      state.recSeconds = Math.floor((Date.now() - recStartTime) / 1000);
      $("rec-timer").textContent = fmtTs(state.recSeconds);
      analyser.getByteFrequencyData(dataArr);
      const avg = dataArr.reduce((a, b) => a + b, 0) / dataArr.length;
      $("rec-meter").style.width = `${Math.min(100, avg * 1.8)}%`;
    }, 150);
  } catch (err) {
    alert(`無法啟動錄音：${err.message}`);
  }
});

$("btn-rec-stop").addEventListener("click", () => {
  clearInterval(state.recInterval);
  $("rec-meter").style.width = "0%";
  $("rec-timer").textContent = "00:00";
  $("btn-rec-start").disabled = false;
  $("btn-rec-stop").disabled = true;
  if (state.recorder && state.recorder.state !== "inactive") {
    state.recorder.stop();
  }
});

// --- Export & Direct Obsidian Save & AI Drawer & Settings ---
document.querySelectorAll("[data-export]").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (!state.currentJobId) return;
    window.location.href = `/api/jobs/${state.currentJobId}/export?fmt=${btn.dataset.export}`;
  });
});

$("btn-save-obsidian").addEventListener("click", async () => {
  if (!state.currentJobId) return;
  const btn = $("btn-save-obsidian");
  btn.disabled = true;
  try {
    const res = await api(`/api/jobs/${state.currentJobId}/export/obsidian`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const openNow = confirm(`✅ 已直接存入 Obsidian Vault！\n路徑：${res.relative_path}\n\n要立即在 Obsidian 中開啟這篇筆記嗎？`);
    if (openNow && res.obsidian_uri) {
      window.location.href = res.obsidian_uri;
    }
  } catch (err) {
    alert(`直存 Obsidian 失敗：${err.message}`);
  } finally {
    btn.disabled = false;
  }
});

$("btn-toggle-ai").addEventListener("click", () => {
  const drawer = $("ai-drawer");
  drawer.style.display = drawer.style.display === "none" ? "flex" : "none";
});
$("btn-close-ai").addEventListener("click", () => ($("ai-drawer").style.display = "none"));

$("btn-copy-prompt").addEventListener("click", async () => {
  if (!state.currentJobId) return;
  const res = await api(`/api/jobs/${state.currentJobId}/summary-prompt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      template: $("summary-template").value,
      custom_prompt: $("summary-custom-prompt").value,
    }),
  });
  await navigator.clipboard.writeText(res.prompt);
  alert("已複製完整「整理 Prompt＋帶人名逐字稿」到剪貼簿！");
});

$("btn-run-summary").addEventListener("click", async () => {
  if (!state.currentJobId) return;
  const btn = $("btn-run-summary");
  btn.disabled = true;
  btn.textContent = "整理中...";
  try {
    const res = await api(`/api/jobs/${state.currentJobId}/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        template: $("summary-template").value,
        custom_prompt: $("summary-custom-prompt").value,
        provider: $("summary-provider").value,
      }),
    });
    $("summary-output").value = res.summary;
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "⚡ AI 生成整理";
  }
});

$("btn-save-summary").addEventListener("click", async () => {
  if (!state.currentJobId) return;
  await api(`/api/jobs/${state.currentJobId}/summary`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ summary: $("summary-output").value }),
  });
  alert("會議摘要已保存！");
});

$("btn-open-settings").addEventListener("click", async () => {
  const cfg = await api("/api/settings");
  $("set-provider").value = cfg.ai_provider;
  $("set-openrouter-key").value = "";
  $("set-openrouter-key").placeholder = cfg.openrouter_configured ? `已設定 (${cfg.openrouter_api_key_masked})` : "未設定";
  $("set-openrouter-model").value = cfg.openrouter_model || "google/gemini-2.5-flash";
  $("set-ollama-url").value = cfg.ollama_url;
  $("set-ollama-model").value = cfg.ollama_model;
  $("set-claude-key").value = "";
  $("set-claude-key").placeholder = cfg.claude_configured ? `已設定 (${cfg.claude_api_key_masked})` : "未設定";
  $("set-claude-model").value = cfg.claude_model;
  $("set-gemini-key").value = "";
  $("set-gemini-key").placeholder = cfg.gemini_configured ? `已設定 (${cfg.gemini_api_key_masked})` : "未設定";
  $("set-gemini-model").value = cfg.gemini_model;
  $("set-obsidian-vault").value = cfg.obsidian_vault_path || "";
  $("set-obsidian-sub").value = cfg.obsidian_subfolder || "20_Areas/工作/會議";
  $("set-hf-token").value = "";
  $("set-hf-token").placeholder = cfg.hf_configured ? `已設定 (${cfg.hf_token_masked})` : "未設定";
  $("set-prompt").value = cfg.default_prompt;
  $("set-glossary").value = Object.entries(cfg.glossary || {})
    .map(([k, v]) => `${k}=${v}`)
    .join("\n");
  $("ollama-detected").textContent = "";
  $("settings-status").textContent = "";
  $("settings-modal").style.display = "flex";
});

$("btn-detect-ollama").addEventListener("click", async () => {
  const info = $("ollama-detected");
  info.textContent = "偵測本機 Ollama 中...";
  const res = await api("/api/ollama/models");
  if (res.available && res.models.length > 0) {
    info.innerHTML = `✅ 已安裝模型：${res.models.map((m) => `<button type="button" class="mini-act ollama-pick" data-m="${escapeHtml(m)}">${escapeHtml(m)}</button>`).join(" ")}`;
    info.querySelectorAll(".ollama-pick").forEach((b) => {
      b.addEventListener("click", () => ($("set-ollama-model").value = b.dataset.m));
    });
  } else if (res.available) {
    info.textContent = "⚠️ Ollama 有啟動，但尚未下載任何模型（可用 ollama pull qwen2.5:14b）";
  } else {
    info.textContent = "⚠️ 偵測不到本機 Ollama 服務（請確認已開啟 Ollama）";
  }
});

$("btn-close-settings").addEventListener("click", () => ($("settings-modal").style.display = "none"));
$("btn-save-settings").addEventListener("click", async () => {
  const glossary = {};
  $("set-glossary")
    .value.split("\n")
    .forEach((line) => {
      const idx = line.indexOf("=");
      if (idx > 0) {
        const k = line.slice(0, idx).trim();
        const v = line.slice(idx + 1).trim();
        if (k && v) glossary[k] = v;
      }
    });
  await api("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ai_provider: $("set-provider").value,
      openrouter_api_key: $("set-openrouter-key").value,
      openrouter_model: $("set-openrouter-model").value,
      ollama_url: $("set-ollama-url").value,
      ollama_model: $("set-ollama-model").value,
      claude_api_key: $("set-claude-key").value,
      claude_model: $("set-claude-model").value,
      gemini_api_key: $("set-gemini-key").value,
      gemini_model: $("set-gemini-model").value,
      obsidian_vault_path: $("set-obsidian-vault").value,
      obsidian_subfolder: $("set-obsidian-sub").value,
      hf_token: $("set-hf-token").value,
      default_prompt: $("set-prompt").value,
      glossary,
    }),
  });
  $("summary-provider").value = $("set-provider").value;
  state.pipelineCfg.prompt = $("set-prompt").value;
  $("settings-status").textContent = "✅ 設定與字典已保存（金鑰僅存於本機 settings.json）";
});

function escapeHtml(str) {
  return String(str || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

loadCatalogAndDefaults();
loadJobs();
