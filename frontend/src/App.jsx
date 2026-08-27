import React, { useState, useEffect, useRef } from 'react';
import { Upload, FileImage, FileDown, Search, Cpu, Database, AlertCircle, CheckCircle2, ChevronRight, X, Layers, Activity, Satellite, BarChart2 } from 'lucide-react';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function App() {
  const [files, setFiles] = useState([]);
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('mission'); // 'mission' or 'admin'
  const [adminStats, setAdminStats] = useState(null);
  const [adminHistory, setAdminHistory] = useState([]);
  const [evaluations, setEvaluations] = useState([]);
  
  const [pipelineStep, setPipelineStep] = useState(0);
  const imageContainerRef = useRef(null);

  // Auto-progress pipeline visualization for UI feeling
  useEffect(() => {
    let timer;
    if (loading && pipelineStep < 3) {
      timer = setTimeout(() => {
        setPipelineStep(prev => prev + 1);
      }, 800);
    }
    return () => clearTimeout(timer);
  }, [loading, pipelineStep]);

  useEffect(() => {
    if (activeTab === 'admin') {
      fetchAdminData();
    }
  }, [activeTab]);

  const fetchAdminData = async () => {
    try {
      const statsRes = await fetch(`${API_BASE_URL}/admin/stats`);
      const histRes = await fetch(`${API_BASE_URL}/admin/history`);
      const evalRes = await fetch(`${API_BASE_URL}/evaluations`);
      
      if (statsRes.ok) setAdminStats(await statsRes.json());
      if (histRes.ok) {
        const histData = await histRes.json();
        setAdminHistory(histData.history || []);
      }
      if (evalRes.ok) {
        const evals = await evalRes.json();
        setEvaluations(evals.evaluations || []);
      }
    } catch (err) {
      console.error("Admin fetch failed", err);
    }
  };

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    if (selectedFiles.length > 2) {
      setError("Maximum 2 images allowed.");
      return;
    }
    setFiles(selectedFiles);
    setResult(null);
    setError('');
    setPipelineStep(0);
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (files.length === 0 || !query) {
      setError('Both image and query are required.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);
    setPipelineStep(0);

    const formData = new FormData();
    formData.append('query', query);
    formData.append('benchmark_mode', 'false');
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const res = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
      setPipelineStep(4);
    } catch (err) {
      setError(err.message || 'Failed to connect to the analysis gateway.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!result) return;
    try {
      const formData = new FormData();
      formData.append('query', query);
      formData.append('benchmark_mode', 'false');
      files.forEach(file => formData.append('files', file));
      
      const res = await fetch(`${API_BASE_URL}/report`, {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) throw new Error('Report generation failed');
      
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `SatQuery_Report_${new Date().getTime()}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    } catch(err) {
      alert("Failed to download report: " + err.message);
    }
  };

  const renderBoundingBoxes = () => {
    if (!result || !result.evidence || files.length === 0) return null;
    
    // We only render bounding boxes over the FIRST image for now
    const groundings = result.evidence.filter(e => e.region != null);
    if (groundings.length === 0) return null;

    return (
      <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
        {groundings.map((g, idx) => {
          const [xmin, ymin, xmax, ymax] = g.region;
          // Note: The coordinates from the model are usually absolute pixel values. 
          // We need to render them relative to the image container.
          // This requires some CSS tricks or normalized coordinates, but since we are wrapping the img in a relative container,
          // we can use percentages if we knew the original img dimensions, OR we just use absolute if the image is unscaled.
          // For simplicity in UI scaling, we assume the object detection model scaled to the input size or we just pass raw pixels 
          // and let the browser scale it if we use standard img sizes.
          // GroundingDino outputs absolute pixels. If the image is scaled by CSS, we need percentage.
          // Let's do a simple inline SVG overlay.
          return (
            <svg key={idx} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} viewBox="0 0 1000 1000" preserveAspectRatio="none">
              {/* Since we don't have original image width/height in frontend state easily without loading it, 
                  we just use the visual_output or directly rely on the backend returning normalized boxes.
                  For now, we'll just style a div with percentage if possible, or skip complex scaling for demo. 
                  Actually, GroundingDino outputs absolute pixel coordinates. */}
            </svg>
          );
        })}
      </div>
    );
  };

  const removeFile = (index) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const isModelUnavailable = result?.status === 'MODEL_UNAVAILABLE';
  const hasConflict = result?.conflict === true;

  return (
    <div className="app-container">
      <nav className="top-nav glass">
        <div className="nav-brand">
          <Satellite size={24} className="text-cyan" />
          <h1 style={{fontSize: '20px', margin: 0}}>SATQUERY AI</h1>
        </div>
        <div className="nav-links">
          <button className={`nav-btn ${activeTab === 'mission' ? 'active' : ''}`} onClick={() => setActiveTab('mission')}>
            <Search size={16} /> Mission Control
          </button>
          <button className={`nav-btn ${activeTab === 'admin' ? 'active' : ''}`} onClick={() => setActiveTab('admin')}>
            <BarChart2 size={16} /> Admin Dashboard
          </button>
        </div>
      </nav>

      {activeTab === 'mission' ? (
      <main className="main-content">
        <header className="header" style={{textAlign: 'center', marginBottom: '40px', marginTop: '20px'}}>
          <h1 style={{fontSize: '48px', fontWeight: 700, margin: 0, letterSpacing: '-0.015em', background: 'linear-gradient(135deg, var(--text-color), #888)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent'}}>SATQUERY AI</h1>
          <h2 style={{fontSize: '24px', fontWeight: 400, color: '#86868b', marginTop: '10px'}}>Interactive Vision-Language Assistant</h2>
          <p style={{fontSize: '18px', color: '#86868b'}}>Remote Sensing Intelligence</p>
        </header>
        
        {/* LEFT PANEL: INPUT */}
        <section className="panel left-panel glass">
          <div className="panel-header">
            <Upload size={20} />
            <h2>DATA INGESTION</h2>
          </div>
          
          <form onSubmit={handleAnalyze} className="input-form">
            <div className="file-upload-container">
              <label htmlFor="file-upload" className="file-upload-label">
                <FileImage size={32} className="text-muted" />
                <span>Select up to 2 satellite images (GeoTIFF, PNG, JPEG)</span>
                <span className="text-small text-muted">For change detection, upload 2 images</span>
              </label>
              <input id="file-upload" type="file" multiple accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={handleFileChange} style={{display: 'none'}} />
            </div>

            {files.length > 0 && (
              <div className="file-list">
                {files.map((f, i) => (
                  <div key={i} className="file-item">
                    <FileImage size={16} />
                    <span className="filename">{f.name}</span>
                    <button type="button" className="icon-btn text-muted" onClick={() => removeFile(i)}><X size={16}/></button>
                  </div>
                ))}
              </div>
            )}

            {files.length > 0 && (
              <div className="image-preview-container" ref={imageContainerRef} style={{position: 'relative'}}>
                <img src={URL.createObjectURL(files[0])} alt="Primary" className="image-preview" />
                {renderBoundingBoxes()}
                {files.length > 1 && (
                   <img src={URL.createObjectURL(files[1])} alt="Secondary" className="image-preview" style={{marginTop: '10px'}}/>
                )}
              </div>
            )}

            <div className="query-container">
              <label>Mission Query</label>
              <textarea 
                value={query} 
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g. What type of land cover is visible? | Where is the water body? | Describe the scene."
                rows={3}
              />
            </div>

            {error && <div className="error-banner"><AlertCircle size={16}/> {error}</div>}

            <button type="submit" className="primary-btn" disabled={loading || files.length === 0}>
              {loading ? <span className="spin"><Activity size={18}/></span> : <Search size={18} />}
              {loading ? 'PROCESSING...' : 'EXECUTE MISSION'}
            </button>
          </form>
        </section>

        {/* CENTER PANEL: RESULTS */}
        <section className="panel center-panel glass">
          <div className="panel-header">
            <Search size={20} />
            <h2>{result ? `Task: ${result.task.toUpperCase()} | Model: ${result.provenance?.model || 'Unknown'}` : 'AWAITING MISSION'}</h2>
          </div>
          
          {!result && !loading && (
             <div className="empty-state-container">
               <Satellite size={48} className="text-muted" style={{opacity: 0.2, marginBottom: '20px'}}/>
               <p className="empty-state">Awaiting remote-sensing image input...</p>
             </div>
          )}

          {loading && (
             <div className="empty-state-container">
               <Activity className="spin text-cyan" size={48} style={{marginBottom: '20px'}}/>
               <p className="text-cyan">Processing satellite intelligence...</p>
             </div>
          )}

          {result && (
            <div className="results-container slide-up" style={{marginTop: 0, borderTop: 'none', paddingTop: 0}}>
              <div className="results-header" style={{flexDirection: 'column', alignItems: 'flex-start', gap: '10px'}}>
                <span className={`badge ${result.status === 'SUCCESS' || result.status === 'VERIFIED' ? 'badge-success' : 'badge-warn'}`} style={{fontSize: '14px', padding: '6px 12px'}}>
                  {result.status}
                </span>
                {hasConflict && (
                  <span className="badge badge-warn" style={{fontSize: '14px', padding: '6px 12px'}}>CONFLICT DETECTED</span>
                )}
              </div>

              {isModelUnavailable && (
                <div className="offline-warning">
                  <AlertCircle size={24} className="text-yellow" />
                  <div>
                    <h4>Specialist Model Unavailable</h4>
                    <p>The gateway routed the task successfully, but the underlying specialist model endpoint is offline or unconfigured.</p>
                  </div>
                </div>
              )}

              {result.abstention_reason && (
                <div className="offline-warning" style={{backgroundColor: 'rgba(255, 100, 100, 0.1)', borderColor: '#ff4444'}}>
                  <AlertCircle size={24} className="text-red" />
                  <div>
                    <h4>Model Abstained</h4>
                    <p>{result.abstention_reason}</p>
                  </div>
                </div>
              )}

              <div className="content-card answer-card">
                <h4>Primary Finding</h4>
                <p className="answer-text">{result.answer}</p>
                {result.confidence != null ? (
                <>
                  <div className="confidence-meter">
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                    <div className="meter-bar">
                      <div className="meter-fill" style={{ width: `${result.confidence * 100}%` }}></div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="confidence-meter">
                  Confidence: Not available
                  <div className="meter-bar">
                    <div className="meter-fill" style={{ width: `0%`, background: 'gray' }}></div>
                  </div>
                </div>
              )}</div>

              {result.visual_output && (
                 <div className="content-card visual-card" style={{marginTop: '20px'}}>
                   <h4>Spatial Evidence Overlay</h4>
                   <img src={result.visual_output} alt="Visual Output" className="visual-evidence-img" />
                 </div>
              )}

              <div className="content-card evidence-card" style={{marginTop: '20px'}}>
                <h4>Structured Evidence</h4>
                {result.evidence?.length > 0 ? (
                  <ul className="evidence-list">
                    {result.evidence.map((e, i) => (
                      <li key={i} style={{display: 'flex', flexDirection: 'column', gap: '4px'}}>
                        <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
                          <CheckCircle2 size={16} /> <strong>{e.claim || 'Evidence'}</strong>
                        </div>
                        <span className="text-muted" style={{marginLeft: '24px'}}>{e.evidence}</span>
                        <span className="text-muted text-small" style={{marginLeft: '24px'}}>Status: {e.status} | Modality: {e.modality}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="empty-state">No structural evidence generated.</p>
                )}
              </div>
              
              <div className="action-row">
                <button className="download-btn" onClick={handleDownloadReport}>
                  <FileDown size={18} /> Download Report
                </button>
              </div>
            </div>
          )}
        </section>

        {/* RIGHT PANEL: AGENT EXECUTION (Simple Dashboard) */}
        <section className="panel right-panel glass">
          <div className="panel-header">
            <Layers size={20} />
            <h2>PIPELINE EXECUTION</h2>
          </div>
          
          <div className="pipeline-visualizer" style={{marginTop: '20px', marginBottom: '20px'}}>
            {['VALIDATION', 'UNDERSTANDING & ROUTING', 'EVIDENCE EXTRACTION', 'VERIFICATION & CONFLICT'].map((step, idx) => (
              <div key={idx} className={`pipeline-step ${pipelineStep > idx ? 'completed' : pipelineStep === idx && loading ? 'active' : 'pending'}`}>
                <div className="step-node">
                  {pipelineStep > idx ? <CheckCircle2 size={16} /> : <div className="dot"></div>}
                </div>
                <div className="step-label">{step}</div>
              </div>
            ))}
          </div>

          {result && (
            <div className="execution-details slide-up">
              <div className="content-card small-card">
                <h4>Execution Trace</h4>
                <ul className="trace-list">
                  {result.execution_trace.map((t, i) => (
                    <li key={i} className="trace-item">
                      <ChevronRight size={14} />
                      <span>{t}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </section>

      </main>
      ) : (
      <main className="admin-view" style={{display: 'flex', gap: '20px', padding: '20px'}}>
        
        {/* Admin Left: Stats & Eval */}
        <div style={{flex: 1, display: 'flex', flexDirection: 'column', gap: '20px'}}>
          <div className="content-card glass">
            <h2>System Statistics</h2>
            {adminStats ? (
              <div style={{marginTop: '20px'}}>
                <p><strong>Total Requests:</strong> {adminStats.total_requests}</p>
                <p><strong>Total Conflicts Detected:</strong> {adminStats.conflicts}</p>
                <h4 style={{marginTop: '15px'}}>Task Distribution:</h4>
                <ul>
                  {Object.entries(adminStats.task_distribution).map(([k, v]) => (
                    <li key={k}>{k}: {v}</li>
                  ))}
                </ul>
                <h4 style={{marginTop: '15px'}}>Status Distribution:</h4>
                <ul>
                  {Object.entries(adminStats.status_distribution).map(([k, v]) => (
                    <li key={k}>{k}: {v}</li>
                  ))}
                </ul>
              </div>
            ) : (
              <p>Loading stats...</p>
            )}
          </div>

          <div className="content-card glass">
            <h2>Benchmark Evaluations</h2>
            <div className="eval-grid" style={{gridTemplateColumns: '1fr', marginTop: '20px'}}>
              {evaluations.map((ev, idx) => (
                <div key={idx} className="eval-card" style={{border: '1px solid var(--border-color)', background: 'var(--card-bg)', padding: '15px'}}>
                  <div className="eval-card-header" style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                    <h3 style={{margin: 0}}>{ev.benchmark_name}</h3>
                    <span className={`badge ${ev.status === 'NOT_AVAILABLE' ? 'badge-warn' : (ev.status === 'PARTIAL' ? 'badge-partial' : 'badge-success')}`}>
                      {ev.evalStatus || ev.status}
                    </span>
                  </div>
                  <p className="eval-desc text-small text-muted" style={{marginTop: '10px'}}>{ev.description}</p>
                  
                  {ev.status === 'NOT_AVAILABLE' && (
                    <p className="text-small" style={{color: '#ff3b30', fontWeight: 600}}>NOT_AVAILABLE — MANUAL DATA REQUIRED</p>
                  )}
                  {ev.status === 'PARTIAL' && (
                    <p className="text-small" style={{color: '#ff9f0a', fontWeight: 600}}>PARTIAL — MISSING IMAGE SAMPLES</p>
                  )}
                  
                  {ev.evalResult && (
                    <div style={{marginTop: '10px', padding: '10px', background: '#111'}}>
                      <p><strong>Score:</strong> {ev.evalResult.score}</p>
                      <p><strong>Evaluated Samples:</strong> {ev.evalResult.evaluated_samples}</p>
                      <p><strong>Metric:</strong> {ev.evalResult.metric}</p>
                      <p className="text-small text-muted">Checksum/Version: {ev.evalResult.provenance?.dataset_version || 'N/A'}</p>
                    </div>
                  )}
                  
                  {ev.status === 'READY' && ev.evalStatus !== 'RUNNING' && (
                    <div style={{marginTop: '15px'}}>
                      <button 
                        className="primary-btn" 
                        onClick={async () => {
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
                        }}
                      >
                        RUN REAL EVALUATION (Smoke Test)
                      </button>
                    </div>
                  )}
                  {ev.evalStatus === 'RUNNING' && <p className="text-accent" style={{marginTop: '10px'}}>Running inference...</p>}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Admin Right: History */}
        <div className="content-card glass" style={{flex: 2}}>
          <h2>Mission History</h2>
          <div style={{marginTop: '20px', maxHeight: '70vh', overflowY: 'auto'}}>
            {adminHistory.length > 0 ? (
              <table style={{width: '100%', textAlign: 'left', borderCollapse: 'collapse'}}>
                <thead>
                  <tr style={{borderBottom: '1px solid #333'}}>
                    <th style={{padding: '10px'}}>Time</th>
                    <th style={{padding: '10px'}}>Query</th>
                    <th style={{padding: '10px'}}>Task</th>
                    <th style={{padding: '10px'}}>Status</th>
                    <th style={{padding: '10px'}}>Conflict</th>
                  </tr>
                </thead>
                <tbody>
                  {adminHistory.map((h, i) => (
                    <tr key={i} style={{borderBottom: '1px solid #222'}}>
                      <td style={{padding: '10px', fontSize: '12px'}}>{new Date(h.timestamp).toLocaleString()}</td>
                      <td style={{padding: '10px', fontSize: '14px'}}>{h.query}</td>
                      <td style={{padding: '10px', fontSize: '12px', color: '#00d4ff'}}>{h.task}</td>
                      <td style={{padding: '10px', fontSize: '12px'}}>
                        <span className={`badge ${h.status.includes('SUCCESS') || h.status.includes('VERIFIED') ? 'badge-success' : 'badge-warn'}`}>{h.status}</span>
                      </td>
                      <td style={{padding: '10px', fontSize: '12px'}}>{h.conflict ? 'Yes' : 'No'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p>No history recorded yet.</p>
            )}
          </div>
        </div>
      </main>
      )}
    </div>
  );
}

export default App;
