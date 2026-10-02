import React, { useState } from "react";
import "./App.css";


function App() {
  const [formData, setFormData] = useState({
    Ticker: "",
    EPS_est: "",
    TTM_EPS: "",
    PE_Ratio: "",
    Forward_PE: "",
    Open: "",
    High: "",
    Low: "",
    Volume: "",
    daily_change: "",
    w1_change: "",
    m1_change: "",
    m3_change: "",
    y1_change: "",
    wk52_high: "",
    wk52_low: "",
    sma20_change: "",
    sma50_change: "",
    sma200_change: "",
    rsi_14: "",
    volatility_wk: "",
    volatility_1m: "",
    shifted_rep_eps: "",
    surprise_shift1: "",
    high_low_range: "",
    vol_log: "",
    Quarter: "",
    Sector: "",
    Industry: "",
    HeadquartersState: ""
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

      // Convert empty strings to null
    const cleanedData = Object.fromEntries(
      Object.entries(formData).map(([key, value]) =>
        value === "" ? [key, null] : [key, value]
      )
    );

    // ---------- 🟢 Feature Engineering ----------
    // Safely parse numbers
    const openVal   = cleanedData.Open   !== null ? parseFloat(cleanedData.Open)   : null;
    const highVal   = cleanedData.High   !== null ? parseFloat(cleanedData.High)   : null;
    const lowVal    = cleanedData.Low    !== null ? parseFloat(cleanedData.Low)    : null;
    const volVal    = cleanedData.Volume !== null ? parseFloat(cleanedData.Volume) : null;
    const sma20Val  = cleanedData.sma20_change !== null ? parseFloat(cleanedData.sma20_change) : null;

        // high_low_range
    if (openVal !== null && highVal !== null && lowVal !== null && openVal !== 0) {
      cleanedData.high_low_range = (highVal - lowVal) / openVal;
    } else {
      cleanedData.high_low_range = null;
    }

    // open_sma20_ratio (using sma20 if available)
    if (openVal !== null && sma20Val !== null && sma20Val !== 0) {
      cleanedData.open_sma20_ratio = openVal / sma20Val;
    } else {
      cleanedData.open_sma20_ratio = null;
    }

    // vol_log
    if (volVal !== null) {
      cleanedData.vol_log = Math.log1p(volVal);
    } else {
      cleanedData.vol_log = null;
    }

    try {
      const response = await fetch("http://192.168.45.248:4200/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cleanedData),
      });
      const data = await response.json();
      console.log("API Response:", data);
      setResult(data.probability.toFixed(3));
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const numericFeatures = [
    "Open", "High", "Low", "Volume", "daily_change", "w1_change", "m1_change", "m3_change", 
    "y1_change", "wk52_high", "wk52_low", "sma20_change", "sma50_change", "sma200_change", 
    "rsi_14", "volatility_wk", "volatility_1m", "EPS_est", "TTM_EPS", "PE_Ratio", "Forward_PE", 
    "shifted_rep_eps", "surprise_shift1", "high_low_range", "vol_log"
  ];
  const categoricalFeatures = ["Ticker", "Quarter", "Sector", "Industry", "HeadquartersState"];

  return (
    <div className="app-container">
      <div className="form-card">
        <h1 className="title">Stock Movement Predictor</h1>
        <form onSubmit={handleSubmit} className="form">
          <h2 className="section-title">Numeric Features</h2>
          <div className="grid">
            {numericFeatures.map((feature) => (
              <input
                key={feature}
                type="number"
                step="any"
                name={feature}
                placeholder={feature}
                value={formData[feature] || ""}
                onChange={handleChange}
                className="input"
              />
            ))}
          </div>

          <h2 className="section-title">Categorical Features</h2>
          <div className="grid">
            {categoricalFeatures.map((feature) => (
              <input
                key={feature}
                type="text"
                name={feature}
                placeholder={feature}
                value={formData[feature] || ""}
                onChange={handleChange}
                className="input"
              />
            ))}
          </div>

          <button type="submit" className="button">Predict</button>
        </form>

        {result && (
          <div className="result">
            <p className="result-text">Probability of Stock Increase:</p>
            <p className="result-value">{result}</p>
          </div>
        )}
      </div>
    </div>
  );

  // return (
  //   <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
  //     <div className="bg-white shadow-xl rounded-lg p-8 max-w-2xl w-full">
  //       <h1 className="text-3xl font-bold text-center mb-6 text-indigo-600">
  //         Stock Movement Predictor
  //       </h1>
  //       <form onSubmit={handleSubmit} className="space-y-6">

  //         {/* Numeric inputs */}
  //         <div>
  //           <h2 className="text-lg font-semibold mb-2">Numeric Features</h2>
  //           <div className="grid grid-cols-2 gap-4">
  //             {numericFeatures.map((feature) => (
  //               <input
  //                 key={feature}
  //                 type="number"
  //                 step="any"
  //                 name={feature}
  //                 placeholder={feature}
  //                 value={formData[feature] || ""}
  //                 onChange={handleChange}
  //                 className="border border-gray-300 rounded-md p-2 w-full"
  //               />
  //             ))}
  //           </div>
  //         </div>

  //         {/* Categorical inputs */}
  //         <div>
  //           <h2 className="text-lg font-semibold mb-2">Categorical Features</h2>
  //           <div className="grid grid-cols-2 gap-4">
  //             {categoricalFeatures.map((feature) => (
  //               <input
  //                 key={feature}
  //                 type="text"
  //                 name={feature}
  //                 placeholder={feature}
  //                 value={formData[feature] || ""}
  //                 onChange={handleChange}
  //                 className="border border-gray-300 rounded-md p-2 w-full"
  //               />
  //             ))}
  //           </div>
  //         </div>

  //         <button
  //           type="submit"
  //           className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-2 px-4 rounded-lg shadow-md transition"
  //         >
  //           Predict
  //         </button>
  //       </form>

  //       {/* Result */}
  //       {result && (
  //         <div className="mt-6 text-center">
  //           <p className="text-xl font-medium text-gray-700">
  //             Probability of Stock Increase:
  //           </p>
  //           <p className="text-3xl font-bold text-green-600">{result}</p>
  //         </div>
  //       )}
  //     </div>
  //   </div>
  // );
}

export default App;