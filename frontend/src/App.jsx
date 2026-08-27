import { useState, useEffect, useRef } from 'react';
import { Upload, Satellite, Target, AlertTriangle, CheckCircle2, ChevronRight, Activity, FileText, X, History, Database, Cpu, Search, Image as ImageIcon } from 'lucide-react';
import './index.css';

const API_BASE_URL = 'http://127.0.0.1:8000';

function App() {
  const [activeTab, setActiveTab] = useState('mission');
  const [images, setImages] = useState([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [pipelineStep, setPipelineStep] = useState(0);
  
  const [adminStats, setAdminStats] = useState(null);
  const [adminHistory, setAdminHistory] = useState([]);
  const [evaluations, setEvaluations] = useState([]);

  useEffect(() => {
    if (activeTab === 'system') {
      fetch(`${API_BASE_URL}/admin/stats`).then(r => r.json()).then(setAdminStats).catch(console.error);
      fetch(`${API_BASE_URL}/admin/history`).then(r => r.json()).then(data => setAdminHistory(data.history || [])).catch(console.error);
      fetch(`${API_BASE_URL}/evaluations`).then(r => r.json()).then(data => setEvaluations(data.evaluations || [])).catch(console.error);
    }
  }, [activeTab]);

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    if (images.length + files.length > 2) {
      alert('Maximum 2 images supported for change/multi-modal analysis.');
      return;
    }
    const newImages = files.map(f => ({
      file: f,
      url: URL.createObjectURL(f)
    }));
    setImages(prev => [...prev, ...newImages]);
  };

  const removeImage = (idx) => {
    setImages(prev => prev.filter((_, i) => i !== idx));
  };

  const setSampleQuery = (q) => {
    setQuery(q);
  };

  const executeMission = async (e) => {
    e.preventDefault();
    if (images.length === 0 || !query) {
      alert('Mission Input Incomplete: Require imagery and query.');
      return;
    }
    
    setLoading(true);
    setResult(null);
    setPipelineStep(0);
    
    const formData = new FormData();
    formData.append('query', query);
    images.forEach((img, idx) => {
      formData.append(idx === 0 ? 'image' : 'image_b', img.file);
    });

    try {
      const interval = setInterval(() => {
        setPipelineStep(s => s < 6 ? s + 1 : s);
      }, 600);

      const res = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });
      
      const data = await res.json();
      clearInterval(interval);
      setPipelineStep(8);
      setResult(data);
    } catch (err) {
      console.error(err);
      alert('Mission Execution Failed. Check telemetry logs.');
      setPipelineStep(0);
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    if (!result) return;
    try {
      const res = await fetch(`${API_BASE_URL}/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(result)
      });
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Mission_Report_${new Date().getTime()}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      alert('Failed to generate telemetry report.');
    }
  };

  return (
    <div className="app-container">
      <nav className="top-nav">
        <div className="nav-brand">
          <Satellite size={18} color="#f5f5f7" />
          <h1>SATQUERY AI</h1>
        </div>
        <div className="nav-links">
          <button className={`nav-btn ${activeTab === 'mission' ? 'active' : ''}`} onClick={() => setActiveTab('mission')}>
            <Target size={14} /> Mission Workspace
          </button>
          <button className={`nav-btn ${activeTab === 'system' ? 'active' : ''}`} onClick={() => setActiveTab('system')}>
            <Activity size={14} /> System Telemetry
          </button>
        </div>
        <div className="system-status">
          <div className="status-indicator"></div>
          SYSTEM ONLINE
        </div>
      </nav>

      <div className="status-strip">
        <div className="status-strip-item">API <span className="status-strip-val">CONNECTED</span></div>
        <div className="status-strip-item">MODEL ENGINE <span className="status-strip-val">READY</span></div>
        <div className="status-strip-item">GPU/CPU <span className="status-strip-val">ACTIVE</span></div>
        <div className="status-strip-item">MISSIONS <span className="status-strip-val">{adminStats ? adminStats.total_requests : 'N/A'}</span></div>
      </div>

      {activeTab === 'mission' ? (
        <main className="main-content fade-in">
          <section className="hero">
            <div className="hero-label">SIH 2026 • SPACE TECHNOLOGY • MISSION CONTROL</div>
            <h1>SATQUERY AI</h1>
            <h2>Interactive Vision-Language Intelligence for Remote Sensing</h2>
            <p>
              Query satellite imagery in natural language. SatQuery AI validates inputs, selects specialist remote-sensing models, extracts evidence, verifies results, and produces an auditable mission report.
            </p>
            <div className="hero-actions">
              <button className="btn-primary" onClick={() => window.scrollTo({top: 800, behavior: 'smooth'})}>Launch Mission</button>
              <button className="btn-secondary" onClick={() => setActiveTab('system')}>Explore Capabilities</button>
            </div>
          </section>

          <div className="workspace">
            <section className="panel left-panel">
              <h3 className="panel-heading">Mission Input</h3>
              <p className="panel-subheading">Upload remote-sensing imagery and define the analytical objective.</p>
              
              <div className="upload-section">
                <input type="file" id="img-upload" multiple accept="image/*,.tif,.tiff" onChange={handleImageUpload} style={{display: 'none'}} />
                <label htmlFor="img-upload" className="upload-area">
                  <Upload size={32} color="#86868b" style={{marginBottom: '16px'}} />
                  <div style={{fontSize: '16px', fontWeight: 600, color: '#f5f5f7'}}>DROP SATELLITE IMAGERY</div>
                  <div style={{fontSize: '12px', color: '#86868b', marginTop: '8px'}}>Supported: GeoTIFF, TIFF, PNG, JPEG (Max 2)</div>
                </label>

                {images.length > 0 && (
                  <div className="image-preview-grid" style={{marginTop: '24px', display: 'flex', gap: '16px'}}>
                    {images.map((img, idx) => (
                      <div key={idx} style={{position: 'relative', width: '120px', height: '120px', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border-color)'}}>
                        <img src={img.url} alt={`Upload ${idx}`} style={{width: '100%', height: '100%', objectFit: 'cover'}} />
                        <div style={{position: 'absolute', bottom: 0, left: 0, right: 0, background: 'rgba(0,0,0,0.7)', padding: '4px', fontSize: '10px', textAlign: 'center'}}>
                          IMAGE {idx === 0 ? 'A' : 'B'}
                        </div>
                        <button onClick={() => removeImage(idx)} style={{position: 'absolute', top: '4px', right: '4px', background: 'rgba(0,0,0,0.5)', border: 'none', borderRadius: '50%', padding: '4px', cursor: 'pointer'}}>
                          <X size={12} color="#fff" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="query-box">
                <h3 className="panel-heading">Natural Language Mission Query</h3>
                <textarea 
                  className="query-input"
                  placeholder="e.g., What changed between these two observations?"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <div className="chips">
                  {['Describe this image', 'Identify the land cover', 'Highlight buildings', 'What changed between these images?', 'Compare optical and SAR imagery'].map((q, i) => (
                    <div key={i} className="chip" onClick={() => setSampleQuery(q)}>{q}</div>
                  ))}
                </div>
              </div>

              <button className="btn-primary execute-btn" onClick={executeMission} disabled={loading}>
                {loading ? 'ANALYZING TELEMETRY...' : 'EXECUTE MISSION'}
              </button>
            </section>

            <section className="panel right-panel">
              <h3 className="panel-heading">Mission Pipeline</h3>
              <p className="panel-subheading">Agent execution trace</p>

              <div className="timeline">
                {[
                  {title: 'INPUT VALIDATION', desc: 'Validating imagery modality and dimensions'},
                  {title: 'QUERY UNDERSTANDING', desc: 'Semantic intent classification'},
                  {title: 'TASK ROUTING', desc: 'Agentic workflow assignment'},
                  {title: 'SPECIALIST SELECTION', desc: 'Loading domain-specific model'},
                  {title: 'EVIDENCE EXTRACTION', desc: 'Running inference'},
                  {title: 'VERIFICATION', desc: 'Cross-checking assertions'},
                  {title: 'CONFLICT ANALYSIS', desc: 'Evaluating logical contradictions'},
                  {title: 'FINAL RESPONSE', desc: 'Generating mission output'}
                ].map((step, idx) => (
                  <div key={idx} className={`timeline-item ${pipelineStep > idx ? 'active' : ''}`}>
                    <div className="timeline-title">
                      {pipelineStep > idx && <CheckCircle2 size={12} style={{marginRight: '6px', color: '#34c759', verticalAlign: 'middle'}}/>}
                      {String(idx + 1).padStart(2, '0')} {step.title}
                    </div>
                    {pipelineStep >= idx && <p className="timeline-desc">{step.desc}</p>}
                  </div>
                ))}
              </div>
            </section>
          </div>

          {result && (
            <div className="workspace fade-in" style={{paddingTop: '0'}}>
              <div className="panel" style={{gridColumn: '1 / -1'}}>
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px'}}>
                  <div>
                    <h3 className="panel-heading">Mission Result</h3>
                    <div style={{display: 'flex', gap: '16px', marginTop: '12px'}}>
                      <div><span style={{fontSize: '10px', color: '#86868b'}}>TASK:</span> <span style={{fontSize: '12px', fontWeight: 600, color: '#fff'}}>{result.task.toUpperCase()}</span></div>
                      <div><span style={{fontSize: '10px', color: '#86868b'}}>STATUS:</span> <span className={`badge ${result.status.includes('VERIFIED') || result.status.includes('SUCCESS') ? 'badge-success' : 'badge-warn'}`}>{result.status}</span></div>
                      <div><span style={{fontSize: '10px', color: '#86868b'}}>CONFIDENCE:</span> <span style={{fontSize: '12px', fontWeight: 600, color: '#fff'}}>{(result.confidence * 100).toFixed(1)}%</span></div>
                    </div>
                  </div>
                  <button className="btn-secondary" onClick={downloadReport} style={{display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', padding: '10px 20px'}}>
                    <FileText size={14} /> DOWNLOAD MISSION REPORT
                  </button>
                </div>

                {result.conflict && (
                  <div style={{background: 'rgba(255, 159, 10, 0.1)', border: '1px solid rgba(255, 159, 10, 0.3)', padding: '16px', borderRadius: '12px', marginBottom: '24px', display: 'flex', gap: '12px', alignItems: 'center'}}>
                    <AlertTriangle color="#ff9f0a" />
                    <div>
                      <div style={{fontSize: '14px', fontWeight: 700, color: '#ff9f0a'}}>CONFLICT DETECTED</div>
                      <div style={{fontSize: '13px', color: '#f5f5f7', marginTop: '4px'}}>{result.conflict_reason}</div>
                    </div>
                  </div>
                )}

                <div className="result-card">
                  <h3>PRIMARY FINDING</h3>
                  <div className="finding">{result.answer}</div>
                </div>

                {result.evidence && result.evidence.length > 0 && (
                  <div className="evidence-panel">
                    <h3 className="panel-heading">Evidence</h3>
                    {result.evidence.map((ev, idx) => (
                      <div key={idx} style={{borderBottom: '1px solid var(--border-color)', paddingBottom: '16px', marginBottom: '16px'}}>
                        <div style={{fontSize: '14px', fontWeight: 600, color: '#fff', marginBottom: '8px'}}>{ev.claim}</div>
                        <div style={{display: 'flex', gap: '16px', fontSize: '12px', color: '#86868b'}}>
                          <span>MODEL: <span style={{color: '#f5f5f7'}}>{ev.source_model}</span></span>
                          <span>MODALITY: <span style={{color: '#f5f5f7'}}>{ev.modality.toUpperCase()}</span></span>
                          <span>CONFIDENCE: <span style={{color: '#f5f5f7'}}>{(ev.confidence * 100).toFixed(1)}%</span></span>
                        </div>
                        {ev.region && <div style={{fontSize: '12px', color: '#0a84ff', marginTop: '8px'}}>Spatial Bounding Box: {JSON.stringify(ev.region)}</div>}
                        {ev.visual_mask_base64 && (
                          <div style={{marginTop: '12px'}}>
                            <div style={{fontSize: '11px', color: '#86868b', marginBottom: '4px'}}>CHANGE MASK OVERLAY</div>
                            <img src={`data:image/png;base64,${ev.visual_mask_base64}`} alt="Mask" style={{maxWidth: '300px', borderRadius: '8px', border: '1px solid var(--border-color)'}} />
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                <div style={{marginTop: '32px'}}>
                  <h3 className="panel-heading">Model & Data Provenance</h3>
                  <table className="provenance-table">
                    <tbody>
                      <tr><th>Model</th><td>{result.provenance?.model || 'N/A'}</td></tr>
                      <tr><th>Dataset</th><td>{result.provenance?.dataset || 'N/A'}</td></tr>
                      <tr><th>Remote-Sensing Adapted</th><td>{result.provenance?.adapted ? 'YES' : 'NO'}</td></tr>
                      <tr><th>Inference Timestamp</th><td>{result.provenance?.timestamp || 'N/A'}</td></tr>
                    </tbody>
                  </table>
                </div>

              </div>
            </div>
          )}
        </main>
      ) : (
        <main className="admin-view fade-in">
          
          <div className="hero" style={{padding: '40px 20px'}}>
            <h1>System Telemetry</h1>
            <h2>Operational Dashboard & Benchmarks</h2>
          </div>

          <div className="workspace" style={{paddingTop: 0}}>
            
            <section className="panel" style={{gridColumn: '1 / -1'}}>
              <h3 className="panel-heading">Benchmark Center</h3>
              <p className="panel-subheading">Real-data evaluation status for SIH requirements</p>
              
              <div className="eval-grid" style={{padding: 0}}>
                {evaluations.map((ev, idx) => (
                  <div key={idx} className="eval-card">
                    <div className="eval-header">
                      <h4 className="eval-title">{ev.benchmark_name}</h4>
                      <span className={`badge ${ev.status === 'NOT_AVAILABLE' ? 'badge-error' : (ev.status === 'PARTIAL' ? 'badge-warn' : 'badge-success')}`}>
                        {ev.evalStatus || ev.status}
                      </span>
                    </div>
                    <p style={{fontSize: '13px', color: '#86868b', margin: '0 0 16px 0', lineHeight: 1.5}}>{ev.description}</p>
                    
                    {ev.status === 'NOT_AVAILABLE' && (
                      <div style={{fontSize: '12px', color: '#ff3b30', fontWeight: 600}}>NOT AVAILABLE — MANUAL DATA REQUIRED</div>
                    )}
                    {ev.status === 'PARTIAL' && (
                      <div style={{fontSize: '12px', color: '#ff9f0a', fontWeight: 600}}>PARTIAL — MISSING IMAGE SAMPLES</div>
                    )}

                    {ev.evalResult && (
                      <div className="eval-stats">
                        <div className="stat-item"><span className="stat-label">Metric</span><span className="stat-val" style={{fontSize: '13px'}}>{ev.evalResult.metric}</span></div>
                        <div className="stat-item"><span className="stat-label">Score</span><span className="stat-val">{ev.evalResult.score}</span></div>
                        <div className="stat-item"><span className="stat-label">Samples</span><span className="stat-val">{ev.evalResult.evaluated_samples}</span></div>
                        <div className="stat-item"><span className="stat-label">Version</span><span className="stat-val" style={{fontSize: '13px'}}>{ev.evalResult.provenance?.dataset_version || 'N/A'}</span></div>
                      </div>
                    )}

                    {ev.status === 'READY' && ev.evalStatus !== 'RUNNING' && (
                      <button className="btn-secondary" style={{width: '100%', marginTop: '24px'}} onClick={async () => {
                        const updated = [...evaluations];
                        updated[idx].evalStatus = 'RUNNING';
                        setEvaluations(updated);
                        try {
                          const res = await fetch(`${API_BASE_URL}/evaluations/${ev.dataset_id}/run?limit=5`, {method: 'POST'});
                          const data = await res.json();
                          const finished = [...evaluations];
                          finished[idx].evalStatus = 'COMPLETED';
                          finished[idx].evalResult = data;
                          setEvaluations(finished);
                        } catch (err) {
                          const failed = [...evaluations];
                          failed[idx].evalStatus = 'FAILED';
                          setEvaluations(failed);
                        }
                      }}>Run Evaluation</button>
                    )}
                  </div>
                ))}
              </div>
            </section>

            <section className="panel" style={{gridColumn: '1 / -1'}}>
              <h3 className="panel-heading">Mission History</h3>
              <div style={{overflowX: 'auto', marginTop: '24px'}}>
                {adminHistory.length > 0 ? (
                  <table className="provenance-table" style={{marginTop: 0}}>
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>Task</th>
                        <th style={{width: 'auto'}}>Query</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {adminHistory.map((h, i) => (
                        <tr key={i}>
                          <td>{new Date(h.timestamp).toLocaleString()}</td>
                          <td style={{color: 'var(--accent-color)', fontWeight: 600}}>{h.task.toUpperCase()}</td>
                          <td style={{fontFamily: 'inherit'}}>{h.query}</td>
                          <td><span className={`badge ${h.status.includes('SUCCESS') || h.status.includes('VERIFIED') ? 'badge-success' : 'badge-warn'}`}>{h.status}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p style={{color: '#86868b', fontSize: '14px'}}>No missions recorded in telemetry.</p>
                )}
              </div>
            </section>

          </div>
        </main>
      )}
    </div>
  );
}

export default App;
