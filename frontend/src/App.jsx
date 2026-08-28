import { useState, useEffect, useRef } from 'react';
import { 
  Upload, Satellite, Target, AlertTriangle, CheckCircle2, 
  Activity, FileText, X, Search, Layers, Database, Map, Box, Info 
} from 'lucide-react';
import './index.css';

const API_BASE_URL = 'http://127.0.0.1:8000';

function App() {
  const [activeTab, setActiveTab] = useState('mission');
  
  // Mission State
  const [images, setImages] = useState([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [pipelineStep, setPipelineStep] = useState(0);
  const [showOverlay, setShowOverlay] = useState(true);
  
  // Telemetry & Benchmark State
  const [adminStats, setAdminStats] = useState(null);
  const [adminHistory, setAdminHistory] = useState([]);
  const [evaluations, setEvaluations] = useState([]);
  const [registryStatus, setRegistryStatus] = useState({});

  useEffect(() => {
    fetch(`${API_BASE_URL}/health`)
      .then(r => r.json())
      .then(d => setRegistryStatus(d.registry_status || {}))
      .catch(console.error);

    fetch(`${API_BASE_URL}/admin/stats`).then(r => r.json()).then(setAdminStats).catch(console.error);
    fetch(`${API_BASE_URL}/admin/history`).then(r => r.json()).then(data => setAdminHistory(data.history || [])).catch(console.error);
    fetch(`${API_BASE_URL}/evaluations`).then(r => r.json()).then(data => setEvaluations(data.evaluations || [])).catch(console.error);
  }, [activeTab]);

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    if (images.length + files.length > 2) {
      alert('Maximum 2 images supported for bi-temporal or multi-modal analysis.');
      return;
    }
    const newImages = files.map(f => ({
      file: f,
      url: URL.createObjectURL(f),
      name: f.name,
      size: (f.size / 1024 / 1024).toFixed(2) + ' MB',
      type: f.name.split('.').pop().toUpperCase()
    }));
    setImages(prev => [...prev, ...newImages]);
  };

  const removeImage = (idx) => {
    setImages(prev => prev.filter((_, i) => i !== idx));
  };

  const buildMissionFormData = () => {
    const formData = new FormData();
    formData.append("query", query);
    formData.append("parameters", JSON.stringify({}));
    formData.append("benchmark_mode", "false");
    images.forEach(img => formData.append("files", img.file));
    return formData;
  };

  const executeMission = async (e) => {
    e.preventDefault();
    if (images.length === 0 || !query) {
      setErrorMsg('MISSION INPUT INCOMPLETE: Imagery and query required.');
      return;
    }
    
    setLoading(true);
    setResult(null);
    setErrorMsg(null);
    setPipelineStep(0);

    const interval = setInterval(() => {
      setPipelineStep(s => s < 6 ? s + 1 : s);
    }, 600);

    try {
      const res = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        body: buildMissionFormData(),
      });
      
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || `HTTP ${res.status}`);
      }

      const data = await res.json();
      clearInterval(interval);
      setPipelineStep(8);
      setResult(data);
    } catch (err) {
      console.error(err);
      clearInterval(interval);
      setPipelineStep(0);
      setErrorMsg(`HTTP ERROR: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    if (images.length === 0 || !query) return;
    try {
      const res = await fetch(`${API_BASE_URL}/report`, {
        method: 'POST',
        body: buildMissionFormData(),
      });
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Mission_Report_${new Date().getTime()}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      alert('Failed to generate telemetry report: ' + err.message);
    }
  };

  const renderNav = () => (
    <nav className="top-nav glass-nav">
      <div className="nav-brand">
        <Satellite size={18} color="#f5f5f7" />
        <h1>SATQUERY AI</h1>
      </div>
      <div className="nav-links">
        {['mission', 'evaluations', 'history', 'system'].map(tab => (
          <button key={tab} className={`nav-btn ${activeTab === tab ? 'active' : ''}`} onClick={() => setActiveTab(tab)}>
            {tab.toUpperCase()}
          </button>
        ))}
      </div>
      <div className="system-status">
        <div className="status-indicator"></div>
        SYSTEM ONLINE
      </div>
    </nav>
  );

  return (
    <div className="app-container">
      {renderNav()}

      {activeTab === 'mission' && (
        <main className="fade-in">
          <section className="hero">
            <h1>SATQUERY AI</h1>
            <h2>Interactive Vision-Language Intelligence for Remote Sensing</h2>
            <div className="hero-label">SIH 2026 • SPACE TECHNOLOGY • MISSION CONTROL</div>
            <p>Query satellite imagery in natural language. Validate. Route. Analyze. Verify. Report.</p>
          </section>

          <div className="mission-workspace">
            {/* LEFT: MISSION INPUT */}
            <section className="panel">
              <div className="panel-header">
                <Upload size={16} /> MISSION INPUT
              </div>
              <p className="panel-subtext">Upload remote-sensing imagery and define the analytical objective.</p>
              
              <div className="upload-section">
                <input type="file" id="img-upload" multiple accept="image/*,.tif,.tiff" onChange={handleImageUpload} style={{display: 'none'}} />
                <label htmlFor="img-upload" className="upload-area">
                  <div style={{fontWeight: 600, color: '#f5f5f7'}}>DROP SATELLITE IMAGERY</div>
                  <div style={{fontSize: '12px', color: '#86868b', marginTop: '4px'}}>GeoTIFF, TIFF, PNG, JPEG (Max 2)</div>
                </label>

                {images.length > 0 && (
                  <div className="image-preview-list">
                    {images.map((img, idx) => (
                      <div key={idx} className="image-preview-item">
                        <img src={img.url} alt={`Upload ${idx}`} />
                        <div className="image-preview-meta">
                          <div className="meta-name">IMAGE {idx === 0 ? 'A' : 'B'} - {img.name}</div>
                          <div className="meta-size">{img.size} • {img.type}</div>
                        </div>
                        <button className="remove-btn" onClick={() => removeImage(idx)}><X size={14} /></button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="query-box">
                <div className="panel-header" style={{marginTop: '24px'}}><Search size={16} /> NATURAL LANGUAGE MISSION QUERY</div>
                <textarea 
                  className="query-input"
                  placeholder="e.g., What changed between these two observations?"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <div className="chips">
                  {['Describe this image', 'What objects are visible?', 'Identify the land cover', 'Highlight buildings', 'What changed between these images?', 'Compare optical and SAR imagery'].map((q, i) => (
                    <div key={i} className="chip" onClick={() => setQuery(q)}>{q}</div>
                  ))}
                </div>
              </div>

              {errorMsg && (
                <div className="error-panel">
                  <AlertTriangle size={16} color="#ff3b30" />
                  <div>
                    <div className="error-title">MISSION EXECUTION FAILED</div>
                    <div className="error-desc">{errorMsg}</div>
                  </div>
                </div>
              )}

              <button className="btn-primary execute-btn" onClick={executeMission} disabled={loading}>
                {loading ? <span className="loading-pulse">ANALYZING...</span> : 'EXECUTE MISSION'}
              </button>
            </section>

            {/* CENTER: SATELLITE VIEWER */}
            <section className="panel viewer-panel">
              <div className="panel-header">
                <Map size={16} /> SATELLITE VIEWER
              </div>
              
              <div className="viewer-container">
                {images.length > 0 ? (
                  <div className="viewer-canvas">
                    {images.map((img, idx) => (
                      <img key={idx} src={img.url} className={`viewer-img ${images.length > 1 && idx === 1 ? 'split-right' : ''}`} alt="Satellite" />
                    ))}
                    
                    {/* Overlays */}
                    {showOverlay && result?.evidence?.map((ev, idx) => {
                      if (ev.region) {
                        const [xmin, ymin, xmax, ymax] = ev.region;
                        // Simplistic relative overlay assuming image fits container. Real impl would map coordinates strictly.
                        return (
                          <div key={idx} className="viewer-bbox" style={{
                            left: `${xmin}px`, top: `${ymin}px`, 
                            width: `${xmax - xmin}px`, height: `${ymax - ymin}px`
                          }}></div>
                        );
                      }
                      if (ev.visual_mask_base64) {
                        return <img key={idx} src={`data:image/png;base64,${ev.visual_mask_base64}`} className="viewer-overlay-img" alt="Change Mask" />;
                      }
                      return null;
                    })}
                  </div>
                ) : (
                  <div className="viewer-empty">NO SENSORY DATA MOUNTED</div>
                )}
                
                <div className="viewer-controls">
                  <button className="viewer-btn" onClick={() => setShowOverlay(!showOverlay)}><Layers size={14}/> OVERLAY</button>
                </div>
              </div>

              {result?.validation && (
                <div className="validation-panel">
                  {Object.entries(result.validation).map(([k, v]) => (
                    <div key={k} className="val-item">
                      <span className="val-key">{k}</span>
                      <span className="val-value">{String(v)}</span>
                    </div>
                  ))}
                </div>
              )}
            </section>

            {/* RIGHT: MISSION INTELLIGENCE */}
            <section className="panel intelligence-panel">
              <div className="panel-header">
                <Target size={16} /> MISSION INTELLIGENCE
              </div>
              
              {/* Timeline */}
              <div className="pipeline-timeline">
                <div className="pipeline-title">MISSION PIPELINE</div>
                <div className="timeline-nodes">
                  {['VALIDATION', 'UNDERSTANDING', 'ROUTING', 'SPECIALIST', 'EVIDENCE', 'VERIFICATION', 'CONFLICT', 'RESPONSE'].map((step, idx) => (
                    <div key={idx} className={`timeline-dot ${pipelineStep > idx ? 'active' : ''}`} title={step}></div>
                  ))}
                </div>
              </div>

              {result ? (
                <div className="intelligence-content fade-in">
                  <div className="result-meta-grid">
                    <div>
                      <div className="meta-label">TASK</div>
                      <div className="meta-val highlight">{result.task.toUpperCase()}</div>
                    </div>
                    <div>
                      <div className="meta-label">STATUS</div>
                      <div className={`badge ${result.status.includes('VERIFIED') || result.status.includes('SUCCESS') ? 'badge-success' : (result.status.includes('UNCERTAIN') ? 'badge-warn' : 'badge-error')}`}>
                        {result.status}
                      </div>
                    </div>
                    <div>
                      <div className="meta-label">CONFIDENCE</div>
                      <div className="meta-val">{result.confidence !== null ? `${(result.confidence * 100).toFixed(1)}%` : 'N/A'}</div>
                    </div>
                  </div>

                  {result.conflict && (
                    <div className="conflict-banner">
                      <AlertTriangle size={14} />
                      <div>
                        <strong>CONFLICT DETECTED</strong>
                        <div>{result.abstention_reason || 'Modality/Spatial Conflict'}</div>
                      </div>
                    </div>
                  )}
                  {result.status === 'UNCERTAIN' && !result.conflict && (
                    <div className="conflict-banner" style={{background: 'rgba(255, 159, 10, 0.1)', color: '#ff9f0a'}}>
                      <Info size={14} /> RESULT REQUIRES REVIEW
                    </div>
                  )}

                  <div className="primary-finding-box">
                    <div className="meta-label">PRIMARY FINDING</div>
                    <div className="finding-text">{result.answer}</div>
                  </div>

                  {result.evidence?.length > 0 && (
                    <div className="evidence-section">
                      <div className="meta-label">EVIDENCE</div>
                      {result.evidence.map((ev, i) => (
                        <div key={i} className="evidence-item">
                          <div className="ev-claim">{ev.claim}</div>
                          <div className="ev-meta">
                            <span>{ev.source_model || 'N/A'}</span> • <span>{ev.modality ? ev.modality.toUpperCase() : 'N/A'}</span> • <span className="highlight">{(ev.confidence * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="provenance-section">
                    <div className="meta-label">MODEL & DATA PROVENANCE</div>
                    <table className="prov-table">
                      <tbody>
                        <tr><td>MODEL</td><td>{result.provenance?.model || 'N/A'}</td></tr>
                        <tr><td>DATASET</td><td>{result.provenance?.dataset || 'N/A'}</td></tr>
                        <tr><td>RS ADAPTED</td><td>{result.provenance?.adapted ? 'YES' : 'N/A'}</td></tr>
                        <tr><td>TIMESTAMP</td><td>{result.provenance?.timestamp || 'N/A'}</td></tr>
                      </tbody>
                    </table>
                  </div>

                  <button className="btn-secondary" onClick={downloadReport} style={{width: '100%', marginTop: '16px'}}>
                    DOWNLOAD MISSION REPORT
                  </button>
                </div>
              ) : (
                <div className="intelligence-empty">AWAITING MISSION EXECUTION</div>
              )}
            </section>
          </div>
        </main>
      )}

      {activeTab === 'evaluations' && (
        <main className="fade-in workspace-single">
          <section className="panel">
            <h2 className="section-title"><Database size={18} /> BENCHMARK CENTER</h2>
            <div className="benchmark-grid">
              {evaluations.map((ev, i) => (
                <div key={i} className="benchmark-card">
                  <div className="bench-header">
                    <h3>{ev.benchmark_name}</h3>
                    <span className={`badge ${ev.status === 'NOT_AVAILABLE' ? 'badge-error' : (ev.status === 'PARTIAL' ? 'badge-warn' : 'badge-success')}`}>
                      {ev.evalStatus || ev.status}
                    </span>
                  </div>
                  <p className="bench-desc">{ev.description}</p>
                  
                  {ev.status === 'NOT_AVAILABLE' && <div className="bench-err">NOT AVAILABLE — MANUAL DATA REQUIRED</div>}
                  {ev.status === 'PARTIAL' && <div className="bench-warn">PARTIAL — MISSING IMAGE SAMPLES</div>}

                  {ev.evalResult && (
                    <div className="bench-stats">
                      <div><div className="stat-lbl">METRIC</div><div className="stat-val">{ev.evalResult.metric}</div></div>
                      <div><div className="stat-lbl">SCORE</div><div className="stat-val">{ev.evalResult.score}</div></div>
                      <div><div className="stat-lbl">SAMPLES</div><div className="stat-val">{ev.evalResult.evaluated_samples}</div></div>
                    </div>
                  )}

                  {ev.status === 'READY' && ev.evalStatus !== 'RUNNING' && (
                    <button className="btn-outline" onClick={async () => {
                      const updated = [...evaluations];
                      updated[i].evalStatus = 'RUNNING';
                      setEvaluations(updated);
                      try {
                        const res = await fetch(`${API_BASE_URL}/evaluations/${ev.dataset_id}/run?limit=5`, {method: 'POST'});
                        if (!res.ok) throw new Error();
                        const data = await res.json();
                        const finished = [...evaluations];
                        finished[i].evalStatus = 'COMPLETED';
                        finished[i].evalResult = data;
                        setEvaluations(finished);
                      } catch (err) {
                        const failed = [...evaluations];
                        failed[i].evalStatus = 'FAILED';
                        setEvaluations(failed);
                      }
                    }}>RUN EVALUATION</button>
                  )}
                </div>
              ))}
            </div>
          </section>
        </main>
      )}

      {activeTab === 'system' && (
        <main className="fade-in workspace-single">
          <section className="panel">
            <h2 className="section-title"><Cpu size={18} /> SYSTEM TELEMETRY</h2>
            <div className="telemetry-grid">
              <div className="telemetry-card">
                <div className="stat-lbl">API CONNECTIVITY</div>
                <div className="stat-val highlight">ONLINE</div>
              </div>
              <div className="telemetry-card">
                <div className="stat-lbl">MODEL ENGINE</div>
                <div className="stat-val highlight">READY</div>
              </div>
              <div className="telemetry-card">
                <div className="stat-lbl">TOTAL MISSIONS</div>
                <div className="stat-val">{adminStats?.total_requests ?? 'N/A'}</div>
              </div>
              <div className="telemetry-card">
                <div className="stat-lbl">CONFLICTS INTERCEPTED</div>
                <div className="stat-val">{adminStats?.conflicts ?? 'N/A'}</div>
              </div>
            </div>

            <h3 className="section-subtitle">MODEL REGISTRY</h3>
            <table className="data-table">
              <thead><tr><th>TASK</th><th>SPECIALIST MODEL</th><th>STATUS</th></tr></thead>
              <tbody>
                <tr><td>VQA</td><td>Salesforce/blip-vqa-base</td><td><span className="badge badge-success">READY</span></td></tr>
                <tr><td>Captioning</td><td>Salesforce/blip-image-captioning-base (RSICD LoRA)</td><td><span className="badge badge-success">READY</span></td></tr>
                <tr><td>Grounding</td><td>IDEA-Research/grounding-dino-base</td><td><span className="badge badge-success">READY</span></td></tr>
                <tr><td>Land Cover</td><td>nielsr/convnext-tiny-finetuned-eurosat</td><td><span className="badge badge-success">READY</span></td></tr>
                <tr><td>Change Map</td><td>Baseline Pixel Difference</td><td><span className="badge badge-success">READY</span></td></tr>
              </tbody>
            </table>
          </section>
        </main>
      )}

      {activeTab === 'history' && (
        <main className="fade-in workspace-single">
          <section className="panel">
            <h2 className="section-title"><History size={18} /> MISSION HISTORY</h2>
            {adminHistory.length > 0 ? (
              <table className="data-table">
                <thead>
                  <tr><th>TIME</th><th>TASK</th><th>QUERY</th><th>STATUS</th><th>CONFIDENCE</th></tr>
                </thead>
                <tbody>
                  {adminHistory.map((h, i) => (
                    <tr key={i}>
                      <td className="tech-font">{new Date(h.timestamp).toLocaleString()}</td>
                      <td className="highlight">{h.task.toUpperCase()}</td>
                      <td>{h.query}</td>
                      <td><span className={`badge ${h.status.includes('SUCCESS') || h.status.includes('VERIFIED') ? 'badge-success' : 'badge-warn'}`}>{h.status}</span></td>
                      <td className="tech-font">{h.confidence !== null ? `${(h.confidence * 100).toFixed(1)}%` : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="empty-state">NO MISSIONS RECORDED</div>
            )}
          </section>
        </main>
      )}

    </div>
  );
}

export default App;
