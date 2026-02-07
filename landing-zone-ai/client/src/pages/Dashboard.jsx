import React, { useState } from 'react'
import { Upload, X, Check, Loader2, AlertCircle, Activity } from 'lucide-react'
import Button from '../components/common/Button'

// Dashboard Specimen - Functional Upload Interface
const Dashboard = () => {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const handleFileChange = (e) => {
    const selected = e.target.files[0]
    if (selected) {
      setFile(selected)
      setPreview(URL.createObjectURL(selected))
      setResult(null) // Reset result
    }
  }

  const handleAnalyze = async () => {
    if (!file) return

    setLoading(true)
    // Simulate API call delay for demo purposes
    // In real implementation: 
    // const formData = new FormData(); formData.append('image', file);
    // await axios.post('http://localhost:5000/predict', formData)
    
    setTimeout(() => {
      setLoading(false)
      // For demo, we just show a success state or a mock result
      // In a real app, 'result' would contain the heatmap URL from backend
      setResult({
        confidence: 0.94,
        uncertainty: 0.02,
        // Using the preview as result for now, but backend would return overlay
        overlayUrl: preview 
      })
    }, 2000)
  }

  return (
    <div className="min-h-screen bg-slate-50 py-12 px-4">
      <div className="container mx-auto max-w-5xl">
        
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Mission Dashboard</h1>
            <p className="text-slate-500">Upload UAV imagery for safety assessment</p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            System Online
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          
          {/* LEFT COLUMN: Upload & Controls */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
              <h3 className="font-semibold text-slate-900 mb-4">Input Source</h3>
              
              {!file ? (
                <div className="border-2 border-dashed border-slate-300 rounded-xl p-8 text-center hover:bg-slate-50 transition-colors cursor-pointer relative">
                  <input 
                    type="file" 
                    accept="image/*" 
                    onChange={handleFileChange}
                    className="absolute inset-0 opacity-0 cursor-pointer"
                  />
                  <div className="w-12 h-12 bg-blue-50 rounded-full flex items-center justify-center mx-auto mb-4 text-blue-600">
                    <Upload className="w-6 h-6" />
                  </div>
                  <p className="text-sm font-medium text-slate-900">Click to upload</p>
                  <p className="text-xs text-slate-500 mt-1">JPG, PNG (Max 10MB)</p>
                </div>
              ) : (
                <div className="relative rounded-xl overflow-hidden border border-slate-200 group">
                  <img src={preview} alt="Input" className="w-full h-48 object-cover" />
                  <button 
                    onClick={() => { setFile(null); setPreview(null); setResult(null); }}
                    className="absolute top-2 right-2 p-1 bg-white/90 rounded-full shadow-sm hover:bg-red-50 text-slate-600 hover:text-red-500 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              )}

              <div className="mt-6">
                <Button 
                  onClick={handleAnalyze} 
                  disabled={!file || loading} 
                  className="w-full justify-center"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Processing...
                    </>
                  ) : (
                    'Run Analysis'
                  )}
                </Button>
              </div>
            </div>

            {/* Model Info Card */}
            <div className="bg-slate-900 text-slate-300 p-6 rounded-2xl">
              <h4 className="text-white font-medium mb-4 flex items-center gap-2">
                <AlertCircle className="w-4 h-4" /> Model Status
              </h4>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span>Architecture</span>
                  <span className="text-white font-mono">YOLOv8-Nano</span>
                </div>
                <div className="flex justify-between">
                  <span>TTA Passes</span>
                  <span className="text-white font-mono">6x</span>
                </div>
                <div className="flex justify-between">
                  <span>Inference Device</span>
                  <span className="text-green-400 font-mono">GPU [Active]</span>
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: Results */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 h-full min-h-[500px] flex flex-col">
              <div className="p-6 border-b border-slate-100 flex justify-between items-center">
                <h3 className="font-semibold text-slate-900">Analysis Results</h3>
                {result && (
                  <span className="px-3 py-1 bg-blue-50 text-blue-700 text-xs font-bold rounded-full uppercase tracking-wide">
                    Analysis Complete
                  </span>
                )}
              </div>
              
              <div className="flex-grow p-6 flex items-center justify-center bg-slate-50/50">
                {!result ? (
                  <div className="text-center text-slate-400">
                    <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Activity className="w-8 h-8 opacity-50" />
                    </div>
                    <p>No analysis data available</p>
                    <p className="text-sm mt-1">Upload an image and run analysis to see heatmaps</p>
                  </div>
                ) : (
                  <div className="w-full h-full flex flex-col">
                    {/* Main Visualizer */}
                    <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-black mb-6 flex-grow">
                      <img src={result.overlayUrl} alt="Result" className="w-full h-full object-contain" />
                      
                      {/* Overlay Labels (Mock) */}
                      <div className="absolute bottom-4 left-4 right-4 flex gap-2 justify-center">
                         <div className="bg-black/70 backdrop-blur text-white px-4 py-2 rounded-lg text-sm">
                            <span className="text-green-400 font-bold">● Green: Safe</span>
                         </div>
                         <div className="bg-black/70 backdrop-blur text-white px-4 py-2 rounded-lg text-sm">
                            <span className="text-red-400 font-bold">● Red: Unsafe</span>
                         </div>
                      </div>
                    </div>

                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-green-50 p-4 rounded-xl border border-green-100">
                        <p className="text-xs text-green-600 font-medium uppercase tracking-wider mb-1">Confidence Score</p>
                        <p className="text-2xl font-bold text-green-700">{(result.confidence * 100).toFixed(1)}%</p>
                      </div>
                      <div className="bg-blue-50 p-4 rounded-xl border border-blue-100">
                        <p className="text-xs text-blue-600 font-medium uppercase tracking-wider mb-1">Uncertainty</p>
                        <p className="text-2xl font-bold text-blue-700">{(result.uncertainty * 100).toFixed(2)}%</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}

export default Dashboard
