import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { 
  Satellite, UploadCloud, X, Send, Activity, ChevronDown, ChevronUp, 
  CheckCircle, AlertCircle, Eye, Cpu, Database, Map, Maximize, Layers, AlertTriangle, FileText, Mic, MicOff
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import './App.css';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('analyze');
  const [health, setHealth] = useState(null);

  // Workspace State
  const [query, setQuery] = useState('');
  const [files, setFiles] = useState([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [execStep, setExecStep] = useState(0);
  const [result, setResult] = useState(null);
  const [traceOpen, setTraceOpen] = useState(false);
  const [isListening, setIsListening] = useState(false);
  
  const startListening = () => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert("Your browser doesn't support speech recognition. Please use Chrome or Edge.");
      return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => setIsListening(true);
    
    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map(result => result[0])
        .map(result => result.transcript)
        .join('');
      setQuery(transcript);
    };

    recognition.onerror = (event) => {
      console.error("Speech error:", event.error);
      setIsListening(false);
    };

    recognition.onend = () => setIsListening(false);
    
    recognition.start();
  };
  
  // Image Preview Refs
  const imageRef = useRef(null);
  const [imgDims, setImgDims] = useState({ w: 0, h: 0 });

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const res = await axios.get(`${API_BASE}/health`);
      setHealth(res.data);
    } catch (e) {
      setHealth({ api_status: 'OFFLINE' });
    }
  };

  const onDrop = useCallback(acceptedFiles => {
    const newFiles = acceptedFiles.map(file => Object.assign(file, {
      preview: URL.createObjectURL(file)
    }));
    setFiles(prev => [...prev, ...newFiles].slice(0, 2)); // Max 2 files
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/tiff': ['.tif', '.tiff'], 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg'] }
  });

  const removeFile = (index) => {
    const newFiles = [...files];
    URL.revokeObjectURL(newFiles[index].preview);
    newFiles.splice(index, 1);
    setFiles(newFiles);
  };

  const handleImageLoad = (e) => {
    if (e.target) {
      setImgDims({ w: e.target.naturalWidth, h: e.target.naturalHeight });
    }
  };

  const runAnalysis = async () => {
    if (!query || files.length === 0) return;
    
    setIsExecuting(true);
    setResult(null);
    setExecStep(1);

    const formData = new FormData();
    formData.append('query', query);
    files.forEach(f => formData.append('files', f));

    let active = true;
    const intervals = [
      { step: 1, delay: 500 },  // Validating inputs
      { step: 2, delay: 1000 }, // Understanding query
      { step: 3, delay: 1500 }, // Selecting tools
      { step: 4, delay: 3000 }, // Executing model
      { step: 5, delay: 3500 }, // Verifying evidence
    ];
    intervals.forEach(({step, delay}) => {
      setTimeout(() => { if(active && isExecuting) setExecStep(step); }, delay);
    });

    try {
      const res = await axios.post(`${API_BASE}/analyze`, formData);
      active = false;
      setExecStep(6);
      setTimeout(() => {
        setResult(res.data);
        setIsExecuting(false);
      }, 500);
    } catch (err) {
      active = false;
      setIsExecuting(false);
      
      let errMsg = `HTTP ERROR: Failed to fetch from backend at ${API_BASE}.`;
      if (err.response) {
        errMsg = `HTTP ERROR ${err.response.status}: ${err.response.data?.detail || err.message}`;
      } else if (err.request) {
        errMsg = `Backend unavailable at ${API_BASE}. ` + 
                 (API_BASE.includes('127.0.0.1') || API_BASE.includes('localhost') 
                  ? "Production frontend is attempting to contact a local backend. A separate heavy-compute server (e.g., Render, GCP, AWS) is required to host the FastAPI inference backend. Once deployed, set VITE_API_BASE_URL in Vercel to point to it." 
                  : "Check if the backend server is running and accessible.");
      }
      
      setResult({
        status: 'BACKEND_UNAVAILABLE',
        answer: errMsg,
        task: 'error',
        execution_trace: ['ERROR: ' + errMsg]
      });
    }
  };

  const generateReport = async () => {
    try {
      const formData = new FormData();
      formData.append('query', query);
      files.forEach(f => formData.append('files', f));
      const res = await axios.post(`${API_BASE}/report`, formData, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'SatQuery_Report.pdf');
      document.body.appendChild(link);
      link.click();
    } catch (e) {
      alert("Report generation failed. Please check backend connection.");
    }
  };

  const getStatusBadge = (status) => {
    if (status === 'SUCCESS' || status === 'VERIFIED') return <span className="status-badge verified"><CheckCircle size={14}/> Evidence Verified</span>;
    if (status === 'PARTIAL' || status === 'PARTIALLY_VERIFIED') return <span className="status-badge partial"><AlertTriangle size={14}/> Partially Verified</span>;
    if (status === 'MODEL_UNAVAILABLE' || status === 'BACKEND_RESOURCE_LIMIT') return <span className="status-badge" style={{background:'rgba(255, 152, 0, 0.15)', color:'#ff9800'}}><AlertTriangle size={14}/> Resource Limit</span>;
    return <span className="status-badge unavailable"><AlertCircle size={14}/> Insufficient Evidence</span>;
  };

  const getToolTypeBadge = (tool, resultObj = null) => {
    if (!tool) return null;
    const baselines = ['change_analysis', 'optical_sar'];
    const isBaseline = baselines.some(b => tool.includes(b));
    if (isBaseline) {
      if (tool.includes('change_analysis')) {
        let isNeural = false;
        if (resultObj && resultObj.evidence) {
           isNeural = resultObj.evidence.some(e => e.model === "ResNet18_Siamese_ImageNet");
        }
        if (isNeural) {
           return (
             <div style={{display:'flex', flexDirection:'column', gap:'4px'}}>
               <span className="status-badge neural" style={{marginBottom: '2px'}}>Pretrained ResNet18 Feature Comparison</span>
             </div>
           );
        }
        return (
          <div style={{display:'flex', flexDirection:'column', gap:'4px'}}>
             <span className="status-badge baseline" style={{marginBottom: '2px'}}>Deterministic Pixel Difference Baseline</span>
          </div>
        );
      }
      if (tool.includes('optical_sar')) return <span className="status-badge baseline">Statistical Baseline</span>;
      return <span className="status-badge baseline">Deterministic Baseline</span>;
    }
    return <span className="status-badge neural">Neural Model</span>;
  };

  // Grounding quality heuristic
  const checkGroundingQuality = (evidence) => {
    if (!evidence || evidence.length === 0) return true;
    for (const ev of evidence) {
      if (ev.region && imgDims.w > 0) {
        const [xmin, ymin, xmax, ymax] = ev.region;
        const areaRatio = ((xmax - xmin) * (ymax - ymin)) / (imgDims.w * imgDims.h);
        if (areaRatio > 0.95) return false; // Covers 95% of image
      }
    }
    return true;
  };

  return (
    <div className="app-container">
      {/* NAVBAR */}
      <nav className="navbar">
        <div className="brand">
          <Satellite size={24} color="var(--accent)" />
          <span className="logo-text">SatQuery AI</span>
          <span className="badge">SIH 2026</span>
        </div>
        <div className="nav-links">
          <span className={`nav-link ${activeTab==='analyze'?'active':''}`} onClick={()=>setActiveTab('analyze')}>Workspace</span>
          <span className={`nav-link ${activeTab==='capabilities'?'active':''}`} onClick={()=>setActiveTab('capabilities')}>Capabilities</span>
        </div>
        <div className="health-status">
          <div className={`dot ${health?.api_status === 'AVAILABLE' || health?.status === 'ok' ? 'online' : 'offline'}`}></div>
          {health?.api_status === 'AVAILABLE' || health?.status === 'ok' ? 'System Nominal' : 'System Offline'}
        </div>
      </nav>

      {/* HERO */}
      {activeTab === 'analyze' && !result && !isExecuting && files.length === 0 && (
        <section className="hero">
          <motion.h1 initial={{opacity:0, y:20}} animate={{opacity:1, y:0}}>Ask Your Satellite Imagery Anything.</motion.h1>
          <motion.p initial={{opacity:0, y:20}} animate={{opacity:1, y:0}} transition={{delay:0.1}}>
            Agentic multimodal remote-sensing analysis powered by vision-language models.
          </motion.p>
        </section>
      )}

      {/* WORKSPACE */}
      {activeTab === 'analyze' && (
        <main className="workspace">
          <div className="workspace-grid" style={{marginTop: files.length > 0 ? '48px' : '0'}}>
            
            {/* LEFT: INPUT */}
            <div className="panel">
              <h2><UploadCloud size={16}/> Input Imagery</h2>
              
              <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
                <input {...getInputProps()} />
                <UploadCloud size={32} className="drop-icon" />
                <div className="drop-text">Drag & drop optical or SAR imagery here</div>
                <div className="drop-sub">Supports single image or temporal pairs (TIFF, PNG, JPEG)</div>
              </div>

              {files.length > 0 && (
                <div className="file-list">
                  {files.length === 2 && (
                    <div style={{fontSize:'11px', color:'var(--accent)', fontWeight:600, marginBottom:'8px'}}>
                      {files[1].name.toLowerCase().includes('sar') ? '✓ Optical + SAR Pair Detected' : '✓ Bi-Temporal Pair Detected'}
                    </div>
                  )}
                  {files.map((file, i) => (
                    <div key={i} className="file-item">
                      <img src={file.preview} className="file-preview" alt="Preview" />
                      <div className="file-info">
                        <div className="file-name">{file.name}</div>
                        <div className="file-meta">
                          <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                          <span>•</span>
                          <span style={{textTransform: 'uppercase'}}>{file.name.toLowerCase().includes('sar') ? 'SAR Data' : 'Optical Data'}</span>
                        </div>
                      </div>
                      <button className="btn-icon" onClick={() => removeFile(i)} aria-label="Remove file"><X size={16}/></button>
                    </div>
                  ))}
                </div>
              )}

              <div className="query-section" style={{marginTop: '32px'}}>
                <h2><Activity size={16}/> Natural Language Query</h2>
                <div className="query-examples">
                  {['Describe this scene', 'Highlight the buildings', 'What changed between these images?', 'Analyze optical and SAR together'].map(q => (
                    <span key={q} className="example-chip" onClick={()=>setQuery(q)}>{q}</span>
                  ))}
                </div>
                <div style={{ position: 'relative' }}>
                  <textarea 
                    className="query-input" 
                    placeholder="E.g., How many buildings are visible?"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                  />
                  <button 
                    onClick={startListening}
                    style={{
                      position: 'absolute',
                      right: '12px',
                      bottom: '12px',
                      background: isListening ? 'var(--accent)' : 'var(--surface-light)',
                      border: 'none',
                      borderRadius: '50%',
                      width: '36px',
                      height: '36px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer',
                      color: isListening ? '#fff' : 'var(--text-muted)'
                    }}
                    title="Speak Query"
                  >
                    {isListening ? <Mic size={18} /> : <MicOff size={18} />}
                  </button>
                </div>
              </div>

              <button className="btn-primary" disabled={!query || files.length === 0 || isExecuting} onClick={runAnalysis}>
                {isExecuting ? 'Executing Analysis...' : 'Analyze Imagery'} <Send size={16} />
              </button>
            </div>

            {/* RIGHT: RESULTS */}
            <div className="panel" style={{minHeight: '600px'}}>
              <h2><Eye size={16}/> SatQuery Intelligence</h2>
              
              {!isExecuting && !result && (
                <div className="empty-state">
                  <Database size={48} />
                  <h3>Ready for satellite analysis</h3>
                  <p>Upload imagery and ask a question to begin.</p>
                </div>
              )}

              {isExecuting && (
                <div className="pipeline-loader">
                  {[
                    {id: 1, label: 'Understanding Query'},
                    {id: 2, label: 'Validating Imagery'},
                    {id: 3, label: 'Selecting Specialists'},
                    {id: 4, label: 'Running Analysis'},
                    {id: 5, label: 'Verifying Evidence'}
                  ].map((step) => (
                    <div key={step.id} className={`pipeline-step ${execStep > step.id ? 'step-done' : execStep === step.id ? 'step-active' : 'step-pending'}`}>
                      <div className="step-icon">
                        {execStep > step.id ? <CheckCircle size={14}/> : <Activity size={14}/>}
                      </div>
                      <div className="step-text">{step.label}</div>
                    </div>
                  ))}
                  <div style={{marginTop:'24px', fontSize:'11px', color:'var(--text-dim)', textAlign:'center'}}>
                    Simulated frontend visualization of SatQuery pipeline state.
                  </div>
                </div>
              )}

              {result && !isExecuting && result.status === 'BACKEND_UNAVAILABLE' && (
                <motion.div initial={{opacity:0}} animate={{opacity:1}} className="unavailable-state" style={{border: '1px solid #d32f2f', background: 'rgba(211, 47, 47, 0.05)'}}>
                  <AlertTriangle size={32} color="#d32f2f" />
                  <h3 style={{color: '#d32f2f'}}>Backend Unavailable</h3>
                  <p>AI backend is currently unavailable.</p>
                  <p style={{marginTop:'12px', fontSize:'12px', color: 'var(--text-muted)'}}>{result.answer}</p>
                  
                  <button className="btn-secondary" style={{marginTop:'24px'}} onClick={()=>setResult(null)}>Try New Query</button>
                </motion.div>
              )}

              {result && !isExecuting && result.status === 'DATA_UNAVAILABLE' && (
                <motion.div initial={{opacity:0}} animate={{opacity:1}} className="unavailable-state">
                  <AlertTriangle size={32} />
                  <h3>Insufficient Evidence</h3>
                  <p>SatQuery could not verify this claim from the supplied imagery.</p>
                  <p style={{marginTop:'12px', fontSize:'12px'}}>Try uploading additional imagery or providing a valid temporal/SAR pair.</p>
                  
                  {result.execution_trace && (
                    <div className="trace-panel" style={{width:'100%', marginTop:'32px', textAlign:'left'}}>
                      <div className="trace-header" onClick={()=>setTraceOpen(!traceOpen)}>
                        <div className="trace-title"><Cpu size={14}/> Agent Execution Trace</div>
                        {traceOpen ? <ChevronUp size={16}/> : <ChevronDown size={16}/>}
                      </div>
                      <AnimatePresence>
                        {traceOpen && (
                          <motion.div initial={{height:0}} animate={{height:'auto'}} exit={{height:0}} style={{overflow:'hidden'}}>
                            <div className="trace-timeline">
                              {result.execution_trace.map((line, i) => {
                                const isAction = line.includes('SUCCESS') || line.includes('FAILED') || line.includes('Selected tools');
                                return (
                                  <div key={i} className="trace-node">
                                    <div className="trace-node-dot"></div>
                                    <div className={`trace-node-content ${!isAction ? 'dim' : ''}`}>{line}</div>
                                  </div>
                                );
                              })}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  )}

                  <button className="btn-secondary" style={{marginTop:'24px'}} onClick={()=>setResult(null)}>Try New Query</button>
                </motion.div>
              )}

              {result && !isExecuting && result.status !== 'DATA_UNAVAILABLE' && result.status !== 'BACKEND_UNAVAILABLE' && (
                <motion.div initial={{opacity:0}} animate={{opacity:1}} className="result-container">
                  
                  <div className="result-header">
                    <div>
                      <div className="result-title">Analysis Complete</div>
                      <div className="result-badges">
                        {getStatusBadge(result.status)}
                      </div>
                    </div>
                  </div>

                  <div className="result-answer">
                    <div style={{fontSize:'10px', color:'var(--text-muted)', letterSpacing:'0.1em', marginBottom:'8px', textTransform:'uppercase'}}>ANSWER</div>
                    {result.answer}
                  </div>

                  {/* VISUAL EVIDENCE PRESENTATION */}
                  {result.task === 'change_analysis' && files.length === 2 ? (
                    <div className="evidence-side-by-side">
                      <div className="side-img-wrapper">
                        <div className="side-img-label">BEFORE</div>
                        <img src={files[0].preview} className="evidence-img" alt="Before" />
                      </div>
                      <div className="side-img-wrapper">
                        <div className="side-img-label">AFTER</div>
                        <img src={files[1].preview} className="evidence-img" alt="After" />
                      </div>
                    </div>
                  ) : result.task === 'optical_sar' && files.length === 2 ? (
                    <div className="evidence-side-by-side">
                      <div className="side-img-wrapper">
                        <div className="side-img-label">OPTICAL</div>
                        <img src={files[0].preview} className="evidence-img" alt="Optical" />
                      </div>
                      <div className="side-img-wrapper">
                        <div className="side-img-label">SAR</div>
                        <img src={files[1].preview} className="evidence-img" alt="SAR" />
                      </div>
                    </div>
                  ) : (
                    <div className="evidence-container">
                      <img 
                        ref={imageRef} 
                        src={files[0].preview} 
                        className="evidence-img" 
                        onLoad={handleImageLoad}
                        alt="Evidence" 
                      />
                      
                      {/* Bounding Box Overlay for Grounding */}
                      {result.evidence?.map((ev, i) => {
                        if (ev.region && imageRef.current && imgDims.w > 0) {
                          const [xmin, ymin, xmax, ymax] = ev.region;
                          const scaleX = imageRef.current.clientWidth / imgDims.w;
                          const scaleY = imageRef.current.clientHeight / imgDims.h;
                          return (
                            <div key={i} className="bounding-box" style={{
                              left: `${xmin * scaleX}px`,
                              top: `${ymin * scaleY}px`,
                              width: `${(xmax - xmin) * scaleX}px`,
                              height: `${(ymax - ymin) * scaleY}px`
                            }}>
                              <div className="bbox-label">Detected Region</div>
                            </div>
                          );
                        }
                        return null;
                      })}
                    </div>
                  )}

                  {!checkGroundingQuality(result.evidence) && (
                    <div className="grounding-warning">
                      <AlertTriangle size={16}/>
                      <div>
                        <strong>Localization quality: Limited.</strong> Zero-shot localization on native Sentinel-2 imagery is currently limited; region bounding spans entire image.
                      </div>
                    </div>
                  )}

                  <div className="meta-grid">
                    <div className="meta-item">
                      <div className="meta-label">Selected Tools</div>
                      <div className="meta-value" style={{textTransform:'uppercase', color:'var(--accent)', display:'flex', flexWrap:'wrap', gap:'8px', marginTop:'4px'}}>
                        {result.selected_tools ? (
                          result.selected_tools.map((t, i) => (
                            <div key={i} style={{display:'flex', flexDirection:'column', gap:'4px'}}>
                              <span>{t}</span>
                              {getToolTypeBadge(t, result)}
                            </div>
                          ))
                        ) : (
                          <div style={{display:'flex', flexDirection:'column', gap:'4px'}}>
                            <span>{result.task}</span>
                            {getToolTypeBadge(result.task, result)}
                          </div>
                        )}
                      </div>
                      {result.selected_tools?.length > 1 && (
                        <div style={{fontSize:'10px', color:'var(--text-muted)', marginTop:'8px'}}>Multi-tool sequential execution</div>
                      )}
                    </div>
                    <div className="meta-item">
                      <div className="meta-label">
                        {result.confidence_type === 'generation_confidence' ? 'Generation Confidence' : 'Model Confidence'}
                      </div>
                      <div className="meta-value" style={{marginTop:'4px'}}>
                        {result.confidence ? (
                          <div>
                            <div>{(result.confidence * 100).toFixed(1)}%</div>
                            {result.confidence_type === 'generation_confidence' && (
                              <div style={{fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px', lineHeight: '1.2'}}>
                                Token-level generation confidence; not factual accuracy.
                              </div>
                            )}
                          </div>
                        ) : 'Model confidence not available'}
                      </div>
                    </div>
                  </div>

                  {result.evidence && result.evidence.length > 0 && (
                    <div className="evidence-panel" style={{marginTop: '24px', background: 'var(--surface-light)', borderRadius: '12px', padding: '16px', border: '1px solid var(--border)'}}>
                      <div style={{fontSize:'12px', fontWeight:'600', color:'var(--text)', marginBottom:'12px', display:'flex', alignItems:'center', gap:'8px'}}>
                        <CheckCircle size={14} color="var(--primary)" /> Extracted Evidence
                      </div>
                      <div style={{display: 'flex', flexDirection: 'column', gap: '12px'}}>
                        {result.evidence.map((ev, i) => (
                          <div key={i} style={{background: 'var(--surface)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-light)'}}>
                            <div style={{fontSize: '13px', fontWeight: '500', color: 'var(--text)', marginBottom: '4px'}}>{ev.claim}</div>
                            <div style={{fontSize: '12px', color: 'var(--text-muted)'}}>{ev.evidence}</div>
                            <div style={{display: 'flex', gap: '8px', marginTop: '8px'}}>
                              <span style={{fontSize: '10px', padding: '2px 6px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', color: 'var(--text-muted)'}}>{ev.model}</span>
                              {ev.confidence && <span style={{fontSize: '10px', padding: '2px 6px', background: 'rgba(76, 175, 80, 0.1)', borderRadius: '4px', color: '#4caf50'}}>Conf: {(ev.confidence*100).toFixed(1)}%</span>}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {result.execution_trace && (
                    <div className="trace-panel">
                      <div className="trace-header" onClick={()=>setTraceOpen(!traceOpen)}>
                        <div className="trace-title"><Cpu size={14}/> Agent Execution Trace</div>
                        {traceOpen ? <ChevronUp size={16}/> : <ChevronDown size={16}/>}
                      </div>
                      <AnimatePresence>
                        {traceOpen && (
                          <motion.div initial={{height:0}} animate={{height:'auto'}} exit={{height:0}} style={{overflow:'hidden'}}>
                            <div className="trace-timeline">
                              {result.execution_trace.map((line, i) => {
                                const isAction = line.includes('SUCCESS') || line.includes('FAILED') || line.includes('Selected tools');
                                return (
                                  <div key={i} className="trace-node">
                                    <div className="trace-node-dot"></div>
                                    <div className={`trace-node-content ${!isAction ? 'dim' : ''}`}>{line}</div>
                                  </div>
                                );
                              })}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  )}

                  <div style={{display:'flex', gap:'16px'}}>
                    <button className="btn-secondary" onClick={()=>setResult(null)}>New Analysis</button>
                    <button className="btn-secondary" onClick={generateReport}><FileText size={16} /> Generate PDF Report</button>
                  </div>

                </motion.div>
              )}
            </div>
          </div>
        </main>
      )}

      {/* CAPABILITIES */}
      {activeTab === 'capabilities' && (
        <main className="workspace" style={{paddingTop: '64px'}}>
          <div className="cap-grid">
            <div className="cap-card">
              <Map size={24} className="cap-icon" />
              <div className="cap-title">Single Image VQA</div>
              <div className="cap-desc">Query optical patches using natural language. Powered by Salesforce BLIP.</div>
              <span className="status-badge neural">Neural Model</span>
            </div>
            <div className="cap-card">
              <Layers size={24} className="cap-icon" />
              <div className="cap-title">RSICD Captioning</div>
              <div className="cap-desc">Specialized domain captioning via custom PyTorch LoRA injection.</div>
              <span className="status-badge neural">Neural Model</span>
            </div>
            <div className="cap-card">
              <Maximize size={24} className="cap-icon" />
              <div className="cap-title">Text-Guided Grounding</div>
              <div className="cap-desc">Localize specific textual classes (e.g., buildings, roads) via GroundingDINO.</div>
              <span className="status-badge neural">Neural Model</span>
            </div>
            <div className="cap-card">
              <Database size={24} className="cap-icon" />
              <div className="cap-title">EuroSAT Land Cover</div>
              <div className="cap-desc">Identify dominant terrain classes using ConvNeXt architectures.</div>
              <span className="status-badge neural">Neural Model</span>
            </div>
            <div className="cap-card">
              <Activity size={24} className="cap-icon" />
              <div className="cap-title">Bi-Temporal Change</div>
              <div className="cap-desc">Analyze pixel-level changes between T1 and T2 timestamps natively.</div>
              <span className="status-badge baseline">Deterministic Baseline</span>
            </div>
            <div className="cap-card">
              <Activity size={24} className="cap-icon" />
              <div className="cap-title">Optical + SAR Analysis</div>
              <div className="cap-desc">Co-register and extract backscatter statistics alongside optical bounds.</div>
              <span className="status-badge baseline">Statistical Baseline</span>
            </div>
          </div>
        </main>
      )}
    </div>
  );
}
