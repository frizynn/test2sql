/* Nivii — Streaming Pipeline Orchestrator (minimalist UI) */

(function () {
  'use strict';

  // --- DOM refs ---
  var page          = document.getElementById('page');
  var form          = document.getElementById('queryForm');
  var input         = document.getElementById('questionInput');
  var submitBtn     = document.getElementById('submitBtn');
  var cancelBtn     = document.getElementById('cancelBtn');

  var progressLine  = document.getElementById('progressLine');
  var progressText  = document.getElementById('progressText');

  var errorDisplay  = document.getElementById('errorDisplay');
  var errorTitle    = document.getElementById('errorTitle');
  var errorDetail   = document.getElementById('errorDetail');

  var resultsArea   = document.getElementById('resultsArea');
  var sqlSection    = document.getElementById('sqlSection');
  var sqlCode       = document.getElementById('sqlCode');
  var sqlTiming     = document.getElementById('sqlTiming');
  var sqlSkeleton   = document.getElementById('sqlSkeleton');
  var copySqlBtn    = document.getElementById('copySqlBtn');
  var detailsSection = document.getElementById('detailsSection');
  var tableSection  = document.getElementById('tableSection');
  var tableHead     = document.getElementById('tableHead');
  var tableBody     = document.getElementById('tableBody');
  var rowCount      = document.getElementById('rowCount');
  var emptyResults  = document.getElementById('emptyResults');
  var tableWrap     = document.getElementById('tableWrap');
  var tableSkeleton = document.getElementById('tableSkeleton');
  var answerSection = document.getElementById('answerSection');
  var answerText    = document.getElementById('answerText');
  var nlgTiming     = document.getElementById('nlgTiming');
  var answerSkeleton = document.getElementById('answerSkeleton');
  var nlgErrorSection = document.getElementById('nlgErrorSection');
  var nlgErrorText  = document.getElementById('nlgErrorText');

  var thinkingBlock  = document.getElementById('thinkingBlock');
  var thinkingToggle = document.getElementById('thinkingToggle');
  var thinkingIcon   = document.getElementById('thinkingIcon');
  var thinkingLabel  = document.getElementById('thinkingLabel');
  var thinkingMeta   = document.getElementById('thinkingMeta');
  var thinkingChevron = document.getElementById('thinkingChevron');
  var traceBody      = document.getElementById('traceBody');

  // --- Abort controller for cancellation ---
  var abortController = null;

  // --- Streaming state ---
  var currentRolloutEl = null;
  var currentChildrenEl = null;
  var rolloutCount = 0;
  var streamStartTime = 0;
  var thinkingStartTime = 0;

  // --- Error message mapping (Spanish, user-friendly) ---
  var ERROR_MESSAGES = {
    model_unavailable: {
      title: 'Servicio no disponible',
      detail: 'El modelo de lenguaje no esta respondiendo. Verifica que el servidor del modelo este activo e intenta de nuevo.',
    },
    model_timeout: {
      title: 'Tiempo de espera agotado',
      detail: 'El modelo tardo demasiado en responder. Intenta de nuevo en unos segundos.',
    },
    model_parse_error: {
      title: 'Respuesta invalida',
      detail: 'El modelo devolvio una respuesta que no se pudo interpretar. Intenta reformular la pregunta.',
    },
    empty_model_output: {
      title: 'Sin resultado',
      detail: 'El modelo no genero una consulta SQL. Intenta reformular la pregunta con mas detalle.',
    },
    max_retries_exceeded: {
      title: 'Consulta fallida',
      detail: 'No se pudo generar una consulta SQL valida despues de varios intentos. Proba con una pregunta mas simple.',
    },
    empty_question: {
      title: 'Pregunta vacia',
      detail: 'Escribi una pregunta antes de consultar.',
    },
    network_error: {
      title: 'Error de conexion',
      detail: 'No se pudo conectar con el servidor. Verifica tu conexion e intenta de nuevo.',
    },
    pipeline_error: {
      title: 'Error en el pipeline',
      detail: 'Ocurrio un error durante el procesamiento. Intenta de nuevo.',
    },
  };

  // --- Helpers ---

  function show(el)  { el.classList.remove('hidden'); }
  function hide(el)  { el.classList.add('hidden'); }

  function setProgressText(text) {
    progressText.textContent = text;
    show(progressLine);
  }

  function resetUI() {
    page.classList.remove('page--has-results');
    hide(progressLine);
    hide(errorDisplay);
    hide(resultsArea);
    hide(answerSection);
    hide(answerSkeleton);
    hide(nlgErrorSection);
    hide(thinkingBlock);
    hide(detailsSection);
    hide(sqlSection);
    hide(sqlSkeleton);
    hide(tableSection);
    hide(tableSkeleton);
    hide(emptyResults);
    show(tableWrap);

    progressText.textContent = '';
    sqlCode.textContent = '';
    sqlTiming.textContent = '';
    tableHead.innerHTML = '';
    tableBody.innerHTML = '';
    rowCount.textContent = '';
    answerText.textContent = '';
    nlgTiming.textContent = '';
    nlgErrorText.textContent = '';
    traceBody.innerHTML = '';
    thinkingLabel.textContent = 'Pensando...';
    thinkingMeta.textContent = '';
    thinkingIcon.classList.remove('is-active');
    thinkingChevron.classList.remove('is-open');
    traceBody.classList.remove('is-open');

    currentRolloutEl = null;
    currentChildrenEl = null;
    rolloutCount = 0;
  }

  function setLoading(loading) {
    input.disabled = loading;
    submitBtn.disabled = loading;
    if (loading) {
      submitBtn.classList.add('is-loading');
      show(cancelBtn);
    } else {
      submitBtn.classList.remove('is-loading');
      hide(cancelBtn);
    }
  }

  function showError(errorCode, serverDetail) {
    var mapped = ERROR_MESSAGES[errorCode] || {
      title: 'Error inesperado',
      detail: serverDetail || 'Ocurrio un error al procesar tu consulta. Intenta de nuevo.',
    };
    errorTitle.textContent = mapped.title;
    errorDetail.textContent = serverDetail || mapped.detail;
    show(errorDisplay);
  }

  function renderTable(columns, rows) {
    if (!columns || columns.length === 0 || !rows || rows.length === 0) {
      hide(tableWrap);
      show(emptyResults);
      rowCount.textContent = '0 filas';
      return;
    }

    var headerRow = document.createElement('tr');
    columns.forEach(function (col) {
      var th = document.createElement('th');
      th.textContent = col;
      headerRow.appendChild(th);
    });
    tableHead.appendChild(headerRow);

    var displayRows = rows.slice(0, 100);
    displayRows.forEach(function (row) {
      var tr = document.createElement('tr');
      row.forEach(function (cell) {
        var td = document.createElement('td');
        td.textContent = cell === null ? '\u2014' : cell;
        tr.appendChild(td);
      });
      tableBody.appendChild(tr);
    });

    rowCount.textContent = rows.length + (rows.length === 1 ? ' fila' : ' filas') +
      (rows.length > 100 ? ' (mostrando 100)' : '');
  }

  function formatMs(ms) {
    if (ms == null) return '';
    if (ms < 1000) return Math.round(ms) + ' ms';
    return (ms / 1000).toFixed(1) + ' s';
  }

  function escHtml(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  // --- Auto-grow textarea ---
  input.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
  });

  // --- Copy SQL button ---
  copySqlBtn.addEventListener('click', function () {
    var sql = sqlCode.textContent;
    if (!sql) return;

    navigator.clipboard.writeText(sql).then(function () {
      copySqlBtn.classList.add('is-copied');
      copySqlBtn.textContent = '\u2713 Copiado';
      setTimeout(function () {
        copySqlBtn.classList.remove('is-copied');
        copySqlBtn.textContent = 'Copiar';
      }, 2000);
    }).catch(function () {
      var range = document.createRange();
      range.selectNodeContents(sqlCode);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    });
  });

  // --- Thinking block helpers ---

  function openThinking() {
    thinkingStartTime = performance.now();
    thinkingLabel.textContent = 'Pensando...';
    thinkingIcon.classList.add('is-active');
    traceBody.classList.add('is-open');
    thinkingChevron.classList.add('is-open');
    show(thinkingBlock);
  }

  function closeThinking() {
    thinkingIcon.classList.remove('is-active');
    var elapsed = ((performance.now() - thinkingStartTime) / 1000).toFixed(0);
    thinkingLabel.textContent = 'Pensamiento';
    thinkingMeta.textContent = elapsed + 's · ' + rolloutCount + ' rollouts';
    traceBody.classList.remove('is-open');
    thinkingChevron.classList.remove('is-open');
  }

  function addTraceStep(iconClass, iconChar, title, detail, sql) {
    var step = document.createElement('div');
    step.className = 'trace-step';
    step.innerHTML =
      '<div class="trace-step__icon ' + iconClass + '">' + iconChar + '</div>' +
      '<div class="trace-step__title">' + escHtml(title) + '</div>' +
      (detail ? '<div class="trace-step__detail">' + escHtml(detail) + '</div>' : '') +
      (sql ? '<div class="trace-step__sql">' + escHtml(sql) + '</div>' : '');
    traceBody.appendChild(step);
    return step;
  }

  function addRolloutStep(rolloutNum) {
    var step = document.createElement('div');
    step.className = 'trace-step';
    step.innerHTML =
      '<div class="trace-step__icon trace-step__icon--refine">\u21BB</div>' +
      '<div class="trace-step__title">Rollout ' + rolloutNum + '</div>';
    traceBody.appendChild(step);

    var childrenContainer = document.createElement('div');
    childrenContainer.className = 'trace-step__children';
    step.appendChild(childrenContainer);

    return { stepEl: step, childrenEl: childrenContainer };
  }

  function addCritiqueToRollout(stepEl, critiqueText) {
    var detailEl = document.createElement('div');
    detailEl.className = 'trace-step__detail';
    detailEl.textContent = critiqueText;
    var title = stepEl.querySelector('.trace-step__title');
    title.insertAdjacentElement('afterend', detailEl);
    title.textContent = title.textContent + ' \u2014 Critique \u2192 Refine';
  }

  function addChildToRollout(childrenEl, childIndex, sql, score, valid, error) {
    var scoreClass = score >= 0 ? 'trace-child__score--positive' : 'trace-child__score--negative';
    var childEl = document.createElement('div');
    childEl.className = 'trace-child';
    childEl.innerHTML =
      (valid ? '\u2705' : '\u274C') + ' Candidato ' + (childIndex + 1) +
      ' \u00B7 <span class="trace-child__score ' + scoreClass + '">' + score + '</span>' +
      (error ? ' \u00B7 ' + escHtml(error) : '') +
      (sql ? '<div class="trace-step__sql">' + escHtml(sql) + '</div>' : '');
    childrenEl.appendChild(childEl);
  }

  // --- Thinking toggle ---
  thinkingToggle.addEventListener('click', function () {
    traceBody.classList.toggle('is-open');
    thinkingChevron.classList.toggle('is-open');
  });

  // --- Cancel handler ---
  cancelBtn.addEventListener('click', function () {
    if (abortController) {
      abortController.abort();
    }
  });

  // --- SSE event handler ---
  function handleSSEEvent(eventType, data, pipelineState) {
    if (eventType === 'step') {
      switch (data.type) {
        case 'direct_generate_start':
          setProgressText('Generando SQL\u2026');
          show(sqlSkeleton);
          show(tableSkeleton);
          break;

        case 'direct_generate_done':
          if (data.status === 'ok' && data.sql) {
            sqlCode.textContent = data.sql;
            pipelineState.sql = data.sql;
            addTraceStep('trace-step__icon--ok', '\u2713', 'Direct Generation', 'SQL generado exitosamente', data.sql);
          } else if (data.status === 'error') {
            addTraceStep('trace-step__icon--error', '\u2717', 'Direct Generation', 'Error: ' + (data.error || 'desconocido'));
          }
          break;

        case 'verifier':
          if (data.status === 'ok') {
            addTraceStep('trace-step__icon--ok', '\u2713', 'Verifier', data.detail || 'Aprobado');
            hide(sqlSkeleton);
            show(sqlSection);
            // Show thinking block collapsed with "fast path" summary
            thinkingLabel.textContent = 'Pensamiento';
            thinkingMeta.textContent = 'fast path';
            show(thinkingBlock);
            pipelineState.verifierPassed = true;
          } else {
            addTraceStep('trace-step__icon--rejected', '\u2717', 'Verifier', data.detail || 'Rechazado');
            setProgressText('Refinando SQL (MCTS)\u2026');
            pipelineState.verifierPassed = false;
          }
          break;

        case 'mcts_start':
          openThinking();
          break;

        case 'rollout_start':
          rolloutCount = data.rollout;
          var rollout = addRolloutStep(data.rollout);
          currentRolloutEl = rollout.stepEl;
          currentChildrenEl = rollout.childrenEl;
          thinkingLabel.textContent = 'Pensando... rollout ' + data.rollout;
          setProgressText('Refinando SQL \u00B7 rollout ' + data.rollout + '\u2026');
          break;

        case 'critique':
          if (currentRolloutEl) {
            addCritiqueToRollout(currentRolloutEl, data.text);
          }
          break;

        case 'child':
          if (currentChildrenEl) {
            addChildToRollout(currentChildrenEl, data.child, data.sql, data.score, data.valid, data.error);
          }
          break;

        case 'best_node':
          if (data.status === 'ok') {
            addTraceStep('trace-step__icon--ok', '\u2605', 'Best Node Selected',
              'Score: ' + data.score + ' \u00B7 ' + data.valid_candidates + ' candidatos validos', data.sql);
            sqlCode.textContent = data.sql;
            pipelineState.sql = data.sql;
          } else {
            addTraceStep('trace-step__icon--error', '\u2717', 'All Nodes Failed',
              'No se pudo producir SQL valido despues de todos los rollouts');
          }
          closeThinking();
          break;
      }

    } else if (eventType === 'result') {
      pipelineState.result = data;
      pipelineState.sql = data.sql;

      hide(sqlSkeleton);
      hide(tableSkeleton);

      // Show answer skeleton at top while NLG runs
      show(answerSkeleton);
      setProgressText('Generando respuesta\u2026');

      // Populate details (SQL + table) — collapsed by default
      sqlCode.textContent = data.sql;
      sqlTiming.textContent = formatMs(performance.now() - streamStartTime);
      show(sqlSection);

      if (data.results) {
        renderTable(data.results.columns, data.results.rows);
      }
      show(tableSection);
      show(detailsSection);

    } else if (eventType === 'error') {
      hide(progressLine);
      hide(sqlSkeleton);
      hide(tableSkeleton);
      hide(answerSkeleton);
      hide(detailsSection);
      hide(thinkingBlock);
      showError(data.error || 'pipeline_error', data.detail);
    }
  }

  // --- NLG call (POST /answer) ---
  async function runNlg(question, sql, results) {
    setProgressText('Generando respuesta\u2026');
    var nlgStart = performance.now();

    try {
      var ansResp = await fetch('/answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question, sql: sql, results: results }),
      });

      var nlgElapsed = performance.now() - nlgStart;

      if (!ansResp.ok) {
        hide(progressLine);
        hide(answerSkeleton);
        var errBody = await ansResp.json().catch(function () {
          return { error: 'model_unavailable', detail: '' };
        });
        var mapped = ERROR_MESSAGES[errBody.error];
        nlgErrorText.textContent = 'No se pudo generar la respuesta en lenguaje natural: ' +
          (mapped ? mapped.detail : errBody.detail || 'Error del modelo de lenguaje.');
        show(nlgErrorSection);
        return;
      }

      var ansData = await ansResp.json();
      hide(progressLine);
      hide(answerSkeleton);

      answerText.textContent = ansData.answer;
      nlgTiming.textContent = formatMs(nlgElapsed);
      show(answerSection);
    } catch (e) {
      console.error('Fetch /answer failed:', e);
      hide(progressLine);
      hide(answerSkeleton);
      nlgErrorText.textContent = 'No se pudo conectar con el servidor para generar la respuesta.';
      show(nlgErrorSection);
    }
  }

  // --- Pipeline execution (streaming) ---

  async function runPipeline(question) {
    resetUI();
    page.classList.add('page--has-results');
    setLoading(true);
    show(resultsArea);

    abortController = new AbortController();
    streamStartTime = performance.now();

    var pipelineState = {
      sql: null,
      result: null,
      verifierPassed: false,
    };

    try {
      var response = await fetch('/ask/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question }),
        signal: abortController.signal,
      });

      if (!response.ok) {
        var errBody = await response.json().catch(function () {
          return { error: 'network_error', detail: '' };
        });
        showError(errBody.error, errBody.detail);
        setLoading(false);
        return;
      }

      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = '';

      while (true) {
        var chunk = await reader.read();
        if (chunk.done) break;

        buffer += decoder.decode(chunk.value, { stream: true });

        var parts = buffer.split('\n\n');
        buffer = parts.pop();

        for (var i = 0; i < parts.length; i++) {
          var part = parts[i];
          if (!part.trim() || part.startsWith(':')) continue;
          var lines = part.split('\n');
          var eventType = 'message';
          var data = '';
          for (var j = 0; j < lines.length; j++) {
            var line = lines[j];
            if (line.startsWith('event: ')) eventType = line.slice(7);
            else if (line.startsWith('data: ')) data += line.slice(6);
          }
          if (data) {
            try {
              handleSSEEvent(eventType, JSON.parse(data), pipelineState);
            } catch (parseErr) {
              console.error('SSE parse error:', parseErr, data);
            }
          }
        }
      }

      if (pipelineState.result) {
        await runNlg(question, pipelineState.result.sql, pipelineState.result.results);
      }

    } catch (e) {
      if (e.name === 'AbortError') {
        hide(progressLine);
        hide(sqlSkeleton);
        hide(tableSkeleton);
        hide(answerSkeleton);
        errorTitle.textContent = 'Consulta cancelada';
        errorDetail.textContent = 'La consulta fue cancelada.';
        show(errorDisplay);
      } else {
        console.error('Streaming failed:', e);
        hide(progressLine);
        hide(sqlSkeleton);
        hide(tableSkeleton);
        hide(answerSkeleton);
        showError('network_error');
      }
    }

    abortController = null;
    setLoading(false);
  }

  // --- Form handler ---
  form.addEventListener('submit', function (e) {
    e.preventDefault();
    var question = input.value.trim();
    if (!question) return;
    runPipeline(question);
  });

  // Enter submits, Shift+Enter adds newline
  input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.dispatchEvent(new Event('submit'));
    }
  });

})();
