// Dashboard.jsx
import React, { useMemo } from "react";
import { useLocation } from "react-router-dom";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  AreaChart,
  Area,
} from "recharts";
import { ParallaxFooter } from "@/components/Footer";
import ScrollToTop from "@/components/ui/scrolltotop";
import { generatePDF } from "../utils/pdfGenerator";

// Risk Tier Colors
const getTierColor = (tier) => {
  if (tier?.includes("Minimal")) return "#6BCB77"; // Green
  if (tier?.includes("Low")) return "#FFD93D";     // Yellow
  if (tier?.includes("Moderate")) return "#FF6B6B"; // Red
  if (tier?.includes("High")) return "#D00000";    // Dark Red
  return "#888888";
};

// Age Category Colors
const getAgeColor = (category) => {
  switch (category) {
    case "Young": return "#4D96FF";       // Blue
    case "Adult": return "#6BCB77";       // Green
    case "Middle Age": return "#FFD93D";  // Yellow
    case "Senior": return "#FF6B6B";      // Red
    default: return "#888888";
  }
};

export default function Dashboard() {
  const location = useLocation();
  const patientData = location.state?.predictions || [];
  const roiData = location.state?.roiData || [];

  if (!patientData.length)
    return <p className="text-center mt-8 text-lg font-semibold">No data available for dashboard.</p>;

  // --- Risk Tier Distribution ---
  const riskTierData = useMemo(() => {
    const tierCount = {};
    patientData.forEach((p) => {
      const tier = p.predictedOutcomes?.[0]?.riskTier || "Unknown";
      tierCount[tier] = (tierCount[tier] || 0) + 1;
    });
    return Object.entries(tierCount).map(([name, value]) => ({ name, value }));
  }, [patientData]);

  // --- Age Category Distribution ---
  const ageCategoryData = useMemo(() => {
    const categories = { "Young": 0, "Adult": 0, "Middle Age": 0, "Senior": 0 };
    patientData.forEach((p) => {
      const age = p.age || 0;
      if (age <= 30) categories["Young"] += 1;
      else if (age <= 50) categories["Adult"] += 1;
      else if (age <= 70) categories["Middle Age"] += 1;
      else categories["Senior"] += 1;
    });
    return Object.entries(categories).map(([name, value]) => ({ name, value }));
  }, [patientData]);

  // --- Top Key Risk Factors ---
  const riskFactorData = useMemo(() => {
    const factorCount = {};
    patientData.forEach((p) => {
      p.predictedOutcomes?.[0]?.keyRiskFactors?.forEach((f) => {
        factorCount[f] = (factorCount[f] || 0) + 1;
      });
    });
    return Object.entries(factorCount)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 10);
  }, [patientData]);

  // --- ROI Data ---
  const roiCostData = useMemo(() => {
    return roiData.map((p) => ({
      patientId: p.patientId,
      proactive: p.predictedCosts?.[0]?.predicted_proactive_cost || 0,
      reactive: p.predictedCosts?.[0]?.predicted_reactive_cost || 0,
      savings: p.predictedCosts?.[0]?.potential_savings || 0,
    }));
  }, [roiData]);

  const roiAgeCategoryData = useMemo(() => {
    const categories = { "Young": [], "Adult": [], "Middle Age": [], "Senior": [] };
    roiData.forEach((p) => {
      const age = p.age_used_for_prediction || 0;
      let category = "";
      if (age <= 30) category = "Young";
      else if (age <= 50) category = "Adult";
      else if (age <= 70) category = "Middle Age";
      else category = "Senior";
      categories[category].push(p.predictedCosts?.[0]?.potential_savings || 0);
    });
    return Object.entries(categories).map(([name, arr]) => ({
      name,
      avgSavings: arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0,
    }));
  }, [roiData]);

  // --- Detailed Report ---
  const generateReport = () => {
    return patientData.map((p, index) => {
      const roi = roiData.find(r => r.patientId === p.patientId);
      const predictedCost = roi?.predictedCosts?.[0] || {};

      return (
        <div key={index} className="bg-white shadow-md rounded-xl p-4 mb-4 border-l-4 border-gradient-to-r from-green-500 to-blue-500">
          <h3 className="font-bold text-lg mb-2">{p.patientId}</h3>
          <p><strong>Age:</strong> {p.age}</p>
          <p><strong>Overall Risk Score:</strong> {p.overallRiskScore}</p>
          <p><strong>Primary Condition:</strong> {p.presentRiskCondition}</p>
          <p><strong>Risk Tier:</strong> {p.predictedOutcomes?.[0]?.riskTier}</p>
          <p><strong>Key Risk Factors:</strong> {p.predictedOutcomes?.[0]?.keyRiskFactors?.join(", ")}</p>
          {predictedCost && (
            <>
              <p><strong>Predicted Proactive Cost:</strong> ${predictedCost.predicted_proactive_cost?.toFixed(2)}</p>
              <p><strong>Predicted Reactive Cost:</strong> ${predictedCost.predicted_reactive_cost?.toFixed(2)}</p>
              <p><strong>Potential Savings:</strong> ${predictedCost.potential_savings?.toFixed(2)}</p>
            </>
          )}
        </div>
      );
    });
  };

  return (
    <div className="p-6 space-y-12 bg-gray-50 min-h-screen">
      <ScrollToTop />
      <h1 className="text-3xl font-bold text-center mb-8 bg-clip-text text-transparent bg-gradient-to-r from-green-500 to-blue-500">
        Patient Risk Dashboard
      </h1>

      {/* Charts Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
        {/* Risk Tier Distribution */}
        <div className="bg-white shadow-lg rounded-2xl p-6 flex flex-col items-center">
          <h2 className="text-xl font-semibold mb-2">Risk Tier Distribution</h2>
          <PieChart width={1000} height={450}>
            <Pie
              data={riskTierData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={120}
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
            >
              {riskTierData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getTierColor(entry.name)} />
              ))}
            </Pie>
            <Tooltip formatter={(value) => `${value} patients`} />
            <Legend verticalAlign="bottom" height={36} />
          </PieChart>
        </div>

        {/* Age Category Distribution */}
        <div className="bg-white shadow-lg rounded-2xl p-6 flex flex-col items-center">
          <h2 className="text-xl font-semibold mb-2">Age Category Distribution</h2>
          <PieChart width={7000} height={350}>
            <Pie
              data={ageCategoryData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={120}
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
            >
              {ageCategoryData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getAgeColor(entry.name)} />
              ))}
            </Pie>
            <Tooltip formatter={(value) => `${value} patients`} />
            <Legend verticalAlign="bottom" height={36} />
          </PieChart>
        </div>
      </div>

      {/* Top Key Risk Factors */}
      <div className="bg-white shadow-lg rounded-2xl p-6 flex flex-col items-center">
        <h2 className="text-xl font-semibold mb-2">Top Key Risk Factors</h2>
        <BarChart width={500} height={350} layout="vertical" margin={{ top: 10, right: 30, left: 120, bottom: 10 }} data={riskFactorData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis dataKey="name" type="category" width={180} />
          <Tooltip />
          <Legend />
          <Bar dataKey="value" fill="#4D96FF" name="Number of Patients" />
        </BarChart>
      </div>

      {/* ROI Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
        {/* ROI Bar Chart */}
        <div className="bg-white shadow-lg rounded-2xl p-6 flex flex-col items-center">
          <h2 className="text-xl font-semibold mb-2">Predicted Costs per Patient</h2>
          <BarChart width={400} height={350} layout="vertical" margin={{ top: 10, right: 30, left: 120, bottom: 10 }} data={roiCostData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="patientId" type="category" width={150} />
            <Tooltip formatter={(value) => `$${value.toFixed(2)}`} />
            <Legend />
            <Bar dataKey="proactive" fill="#6BCB77" name="Proactive Cost" />
            <Bar dataKey="reactive" fill="#4D96FF" name="Reactive Cost" />
          </BarChart>
        </div>

        {/* ROI Area Chart */}
        <div className="bg-white shadow-lg rounded-2xl p-6 flex flex-col items-center">
          <h2 className="text-xl font-semibold mb-2">Proactive vs Reactive Cost Trend</h2>
          <AreaChart width={400} height={350} data={roiCostData} margin={{ top: 10, right: 30, left: 50, bottom: 10 }}>
            <defs>
              <linearGradient id="colorProactive" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6BCB77" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#6BCB77" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorReactive" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#4D96FF" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#4D96FF" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <XAxis dataKey="patientId" />
            <YAxis />
            <CartesianGrid strokeDasharray="3 3" />
            <Tooltip formatter={(value) => `$${value.toFixed(2)}`} />
            <Legend verticalAlign="top" height={36}/>
            <Area type="monotone" dataKey="proactive" stroke="#6BCB77" fillOpacity={1} fill="url(#colorProactive)" name="Proactive Cost" />
            <Area type="monotone" dataKey="reactive" stroke="#4D96FF" fillOpacity={1} fill="url(#colorReactive)" name="Reactive Cost" />
          </AreaChart>
        </div>
      </div>

      {/* ROI: Average Savings by Age Category */}
      <div className="bg-white shadow-lg rounded-2xl p-6 flex flex-col items-center">
        <h2 className="text-xl font-semibold mb-2">Average Potential Savings by Age Group</h2>
        <BarChart width={500} height={350} layout="vertical" margin={{ top: 10, right: 30, left: 120, bottom: 10 }} data={roiAgeCategoryData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis dataKey="name" type="category" width={150} />
          <Tooltip formatter={(value) => `$${value.toFixed(2)}`} />
          <Legend />
          <Bar dataKey="avgSavings" fill="#FFD93D" name="Avg Potential Savings" />
        </BarChart>
      </div>

      {/* Detailed Text Report */}
      <div className="bg-white shadow-lg rounded-2xl p-6">
        <h2 className="text-xl font-semibold mb-4 text-center">Detailed Patient Report</h2>
        {generateReport()}
      </div>

      <ParallaxFooter />
    </div>
  );
}