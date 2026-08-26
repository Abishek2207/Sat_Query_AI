import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { 
  Satellite, Server, AlertCircle, CheckCircle2, 
  UploadCloud, FileImage, X, Search, Activity,
  Database, FileCode2, Info, Map, ChevronRight, Layers, FileDown
} from 'lucide-react';
import './App.css';

const API_BASE_URL = 'http://127.0.0.1:8000';

function App() {
  const [images, setImages] = useState([]);
  const [query, setQuery] = useState('');
  const [benchmarkMode, setBenchmarkMode] = useState(true);
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('ANSWER');
  const [pipelineStep, setPipelineStep] = useState(0);
  const [currentView, setCurrentView] = useState('workspace');
  const [evaluations, setEvaluations] = useState([]);

  const fetchHealth = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/health`);
      setHealth(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchEvaluations = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/evaluations`);
      setEvaluations(res.data.evaluations || []);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchHealth();
    fetchEvaluations();
  }, []);

  // Simulate pipeline progress while loading
  useEffect(() => {
    let interval;
    if (loading) {
      setPipelineStep(1);
      interval = setInterval(() => {
        setPipelineStep(prev => (prev < 4 ? prev + 1 : prev));
      }, 800);
    } else if (result) {
      setPipelineStep(5); // Complete
    } else {
      setPipelineStep(0);
    }
    return () => clearInterval(interval);
  }, [loading, result]);

  const onDrop = useCallback(acceptedFiles => {
    const newFiles = acceptedFiles.map(file => Object.assign(file, {
      preview: URL.createObjectURL(file)
    }));
    setImages(prev => {
      const combined = [...prev, ...newFiles];
      if (combined.length > 2) {
        alert("Maximum 2 images allowed. The first 2 will be kept.");
        return combined.slice(0, 2);
      }
      return combined;
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpeg', '.jpg'],
      'image/png': ['.png'],
      'image/tiff': ['.tif', '.tiff']
    },
    maxFiles: 2
  });

  const removeImage = (idx) => {
    const newImages = [...images];
    URL.revokeObjectURL(newImages[idx].preview);
    newImages.splice(idx, 1);
    setImages(newImages);
  };

  const handleAnalyze = async () => {
    if (images.length === 0) {
      alert("Please upload at least one image.");
      return;
    }
    if (!query) {
      alert("Please enter a query.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    setActiveTab('ANSWER');

    const formData = new FormData();
    formData.append('query', query);
    formData.append('parameters', '{}');
    formData.append('benchmark_mode', benchmarkMode);
    images.forEach(img => {
      formData.append('files', img);
    });

    try {
      const res = await axios.post(`${API_BASE_URL}/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      if (err.response) {
        setError(`Server Error (${err.response.status}): ${JSON.stringify(err.response.data)}`);
      } else if (err.request) {
        setError("Network Error: Could not connect to the backend server at 127.0.0.1:8000.");
      } else {
        setError(`Request Error: ${err.message}`);
      }
    }
    setLoading(false);
  };

  const handleDownloadReport = () => {
    if (!result) return;
    const reportText = `
SATQUERY AI - MISSION REPORT
============================
Timestamp: ${new Date().toISOString()}

[ MISSION DETAILS ]
Task: ${result.task}
Status: ${result.status}
Query: ${query}
Answer: ${result.answer}
Confidence: ${result.confidence ?? 'N/A'}

[ PROVENANCE ]
${result.provenance ? JSON.stringify(result.provenance, null, 2) : 'None'}

[ EXECUTION TRACE ]
${result.execution_trace.join('\n')}

[ VALIDATION ]
${JSON.stringify(result.validation, null, 2)}
    `.trim();
    
    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'SatQuery_Mission_Report.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleSuggestion = (text) => {
    setQuery(text);
  };

  const isModelUnavailable = result && result.status === 'MODEL_UNAVAILABLE';

  return (
    <div className="app-container">
      <div className="orbit-bg"></div>
      
      <header className="top-bar">
        <div className="logo-area">
          <Satellite className="logo-icon" size={32} />
          <div>
            <h1>SATQUERY AI</h1>
            <span className="subtitle">Remote Sensing Intelligence</span>
          </div>
        </div>
        <div className="nav-area">
          <button className={`nav-btn ${currentView === 'workspace' ? 'active' : ''}`} onClick={() => setCurrentView('workspace')}>MISSION WORKSPACE</button>
          <button className={`nav-btn ${currentView === 'evaluations' ? 'active' : ''}`} onClick={() => setCurrentView('evaluations')}>EVALUATION BENCHMARKS</button>
        </div>
        <div className="system-status">
          <div className="status-indicator">
            {health ? <Server className="text-cyan" size={18} /> : <AlertCircle className="text-red" size={18} />}
            <span className={health ? 'text-cyan' : 'text-red'}>
              {health ? 'SYSTEM ONLINE' : 'GATEWAY OFFLINE'}
            </span>
          </div>
        </div>
      </header>

      {currentView === 'workspace' ? (
      <main className="workspace">
        
        {/* LEFT PANEL: INPUT */}
        <section className="panel left-panel glass">
          <div className="panel-header">
            <UploadCloud size={20} />
            <h2>INPUT WORKSPACE</h2>
          </div>
          
          <div className="upload-section">
            <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
              <input {...getInputProps()} />
              <UploadCloud size={32} className="drop-icon" />
              <p>Drag & drop up to 2 images here</p>
              <small>GeoTIFF, TIFF, PNG, JPEG</small>
            </div>

            <div className="image-preview-container">
              {images.map((img, idx) => (
                <div key={idx} className="image-card">
                  <div className="img-thumbnail" style={{backgroundImage: `url(${img.preview})`}}>
                    <button className="remove-btn" onClick={(e) => { e.stopPropagation(); removeImage(idx); }}>
                      <X size={14} />
                    </button>
                  </div>
                  <div className="img-info">
                    <span className="img-name" title={img.name}>{img.name}</span>
                    <span className="img-size">{(img.size / 1024 / 1024).toFixed(2)} MB</span>
                  </div>
                </div>
              ))}
            </div>
            {images.length === 2 && (
              <div className="mode-selector">
                <span className="badge badge-info">Multi-Image Mode</span>
              </div>
            )}
          </div>

          <div className="query-box" style={{marginTop: '20px'}}>
            <label className="input-label">Natural Language Query</label>
            <textarea
              className="query-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="E.g., What is the dominant land cover?"
              rows={3}
            />
            <div className="suggestions" style={{marginTop: '10px', marginBottom: '15px'}}>
              <button className="pill-btn" onClick={() => handleSuggestion("Highlight the water body")}>Grounding</button>
              <button className="pill-btn" onClick={() => handleSuggestion("What changed between these two dates?")}>Change</button>
            </div>
          </div>

          <div className="settings-section" style={{marginTop: '0', marginBottom: '20px'}}>
            <label className="checkbox-label">
              <input 
                type="checkbox" 
                checked={benchmarkMode} 
                onChange={e => setBenchmarkMode(e.target.checked)} 
              />
              <span className="checkbox-text">Benchmark Mode (Allow PNG/JPEG)</span>
            </label>
          </div>

          <button 
            className={`analyze-btn ${loading ? 'loading' : ''}`} 
            onClick={handleAnalyze} 
            disabled={loading || images.length === 0}
          >
            {loading ? <Activity className="spin" size={20} /> : <Search size={20} />}
            <span>{loading ? 'ANALYZING...' : 'EXECUTE MISSION'}</span>
          </button>

          {error && (
            <div className="error-box">
              <AlertCircle size={20} />
              <p>{error}</p>
            </div>
          )}
        </section>

        {/* CENTER PANEL: RESULT */}
        <section className="panel center-panel glass">
          <div className="panel-header">
            <Search size={20} />
            <h2>ANALYSIS RESULT</h2>
          </div>
          
          {!result && !loading && (
             <div className="empty-state-container">
               <Satellite size={48} className="text-muted" style={{opacity: 0.2, marginBottom: '20px'}}/>
               <p className="empty-state">Awaiting mission input...</p>
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
                <span className={`badge ${isModelUnavailable ? 'badge-warn' : result.status === 'SUCCESS' ? 'badge-success' : 'badge-warn'}`} style={{fontSize: '14px', padding: '6px 12px'}}>
                  {result.status}
                </span>
              </div>

              {isModelUnavailable && (
                <div className="offline-warning">
                  <AlertCircle size={24} className="text-yellow" />
                  <div>
                    <h4>Specialist Model Unavailable</h4>
                    <p>The gateway routed the task successfully, but the underlying specialist model endpoint for <b>{result.task}</b> is offline or unconfigured.</p>
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
                   <h4>Spatial Evidence</h4>
                   <img src={result.visual_output} alt="Visual Output" className="visual-evidence-img" />
                 </div>
              )}

              <div className="content-card evidence-card" style={{marginTop: '20px'}}>
                <h4>Supporting Evidence</h4>
                {result.evidence?.length > 0 ? (
                  <ul className="evidence-list">
                    {result.evidence.map((e, i) => <li key={i}><CheckCircle2 size={16} /> {e}</li>)}
                  </ul>
                ) : (
                  <p className="empty-state">No textual evidence provided.</p>
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

        {/* RIGHT PANEL: AGENT EXECUTION */}
        <section className="panel right-panel glass">
          <div className="panel-header">
            <Layers size={20} />
            <h2>AGENT EXECUTION</h2>
          </div>

          <div className="agent-status-box">
             <p className="info-row"><span>Detected Task:</span> <strong className="text-cyan">{result ? result.task : 'N/A'}</strong></p>
          </div>
          
          <div className="pipeline-visualizer" style={{marginTop: '20px', marginBottom: '20px'}}>
            {['VALIDATION', 'TASK ROUTING', 'SPECIALIST MODEL', 'FINAL RESPONSE'].map((step, idx) => (
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

              <div className="content-card small-card" style={{marginTop: '15px'}}>
                <h4>Validation</h4>
                <pre className="code-block mini">{JSON.stringify(result.validation, null, 2)}</pre>
              </div>

              <div className="content-card small-card" style={{marginTop: '15px'}}>
                <h4>Provenance</h4>
                {result.provenance ? (
                  <pre className="code-block mini">{JSON.stringify(result.provenance, null, 2)}</pre>
                ) : (
                  <p className="empty-state">N/A</p>
                )}
              </div>
            </div>
          )}
        </section>

      </main>
      ) : (
      <main className="evaluations-view">
        <div className="eval-header">
          <h2>BENCHMARK EVALUATIONS</h2>
          <p>Scientific integrity requires real data. Fabricated scores are strictly prohibited.</p>
        </div>
        <div className="eval-grid">
          {evaluations.map((ev, idx) => (
            <div key={idx} className="eval-card glass">
              <div className="eval-card-header">
                <h3>{ev.benchmark_name}</h3>
                <span className={`badge ${ev.status === 'NOT_EVALUATED' || ev.status === 'DATASET_NOT_AVAILABLE' ? 'badge-warn' : 'badge-success'}`}>
                  {ev.status}
                </span>
              </div>
              <p className="eval-desc">{ev.description}</p>
              
              <div className="eval-meta">
                <strong>Expected Format:</strong> {ev.expected_format}
              </div>
              
              <div className="eval-meta">
                <strong>Configured Path:</strong> {ev.dataset_path}
              </div>

              {ev.metrics && (
                <div className="eval-metrics">
                  <h4>Metrics:</h4>
                  <pre className="code-block mini">{JSON.stringify(ev.metrics, null, 2)}</pre>
                </div>
              )}
            </div>
          ))}
          
          {evaluations.length === 0 && (
             <div className="empty-state-container">
               <Activity className="spin text-cyan" size={48} style={{marginBottom: '20px'}}/>
               <p className="text-cyan">Fetching benchmark statuses...</p>
             </div>
          )}
        </div>
      </main>
      )}
    </div>
  );
}

export default App;
