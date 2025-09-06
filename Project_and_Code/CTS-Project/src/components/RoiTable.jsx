import React from 'react';
import { TrendingUp, DollarSign, AlertTriangle } from 'lucide-react';

export default function RoiTable({ roiData }) {

  if (!roiData || roiData.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-3xl p-8 shadow-lg mt-10 mb-10">
        <div className="text-center">
          <p className="text-slate-500 text-sm">No ROI data to display. Upload a CSV file to see the analysis.</p>
        </div>
      </div>
    );
  }

  // 🔹 Aggregate costs per patient
  const aggregatedPatients = roiData.map(patient => {
    if (!patient.predictedCosts || patient.predictedCosts.length === 0) {
      return null;
    }

    const earlyTotal = patient.predictedCosts.reduce(
      (sum, c) => sum + (c.predicted_proactive_cost || 0),
      0
    );
    const lateTotal = patient.predictedCosts.reduce(
      (sum, c) => sum + (c.predicted_reactive_cost || 0),
      0
    );
    const savings = patient.predictedCosts.reduce(
      (sum, c) => sum + (c.potential_savings || 0),
      0
    );
    const savingsPercentage =
      lateTotal > 0 ? ((savings / lateTotal) * 100).toFixed(1) : 0;

    return {
      id: patient.patientId,
      name: patient.patientId,
      conditions: patient.predictedCosts.map(c => c.condition).join(', '),
      earlyTotal,
      lateTotal,
      savings,
      savingsPercentage
    };
  }).filter(Boolean);

  // 🔹 Hospital totals
  const hospitalEarlyTotal = aggregatedPatients.reduce((sum, p) => sum + p.earlyTotal, 0);
  const hospitalLateTotal = aggregatedPatients.reduce((sum, p) => sum + p.lateTotal, 0);
  const hospitalSavings = hospitalLateTotal - hospitalEarlyTotal;
  const hospitalSavingsPercentage = hospitalLateTotal > 0
    ? ((hospitalSavings / hospitalLateTotal) * 100).toFixed(1)
    : 0;

  return (
    <div className="bg-white border border-slate-200 rounded-3xl p-8 shadow-lg mt-10 mb-10">
      {/* Header */}
      <div className="flex items-center space-x-4 mb-8">
        <div className="w-12 h-12 bg-gradient-to-br from-green-400 to-blue-400 rounded-xl flex items-center justify-center shadow-md">
          <TrendingUp className="w-6 h-6 text-white" />
        </div>
        <div>
          <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-500 to-blue-500">
            ROI Analysis
          </h2>
          <p className="text-slate-600">Early vs Late Treatment Cost Comparison</p>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden mt-10 mb-10 shadow-md">
        <table className="w-full">
          <thead>
            <tr className="bg-gradient-to-r from-green-50 to-blue-50">
              <th className="px-6 py-4 text-left text-base font-semibold text-green-600">Patient ID</th>
              <th className="px-6 py-4 text-left text-base font-semibold text-green-600">Conditions</th>
              <th className="px-6 py-4 text-left text-base font-semibold text-green-600">Proactive Cost ($)</th>
              <th className="px-6 py-4 text-left text-base font-semibold text-green-600">Reactive Cost ($)</th>
              <th className="px-6 py-4 text-left text-base font-semibold text-green-600">Potential Savings ($)</th>
              {/* <th className="px-6 py-4 text-left text-base font-semibold text-green-600">Savings (%)</th> */}
            </tr>
          </thead>
          <tbody>
            {aggregatedPatients.map((patient) => (
              <tr key={patient.id} className="border-b border-slate-100 hover:bg-green-50/50 transition-colors">
                <td className="px-6 py-4 text-slate-700 font-medium">{patient.name}</td>
                <td className="px-6 py-4 text-slate-700">{patient.conditions}</td>
                <td className="px-6 py-4 text-slate-800 font-semibold">${patient.earlyTotal.toLocaleString()}</td>
                <td className="px-6 py-4 text-red-500 font-semibold">${patient.lateTotal.toLocaleString()}</td>
                <td className="px-6 py-4 text-green-600 font-semibold">${patient.savings.toLocaleString()}</td>
                {/* <td className="px-6 py-4 text-green-600 font-semibold">{patient.savingsPercentage}%</td> */}
              </tr>
            ))}
            {/* Hospital Total Row */}
            <tr className="bg-gradient-to-r from-green-100 to-blue-100 font-bold text-lg">
              <td className="px-6 py-5 text-slate-800" colSpan="2">HOSPITAL TOTAL</td>
              <td className="px-6 py-5 text-slate-800">${hospitalEarlyTotal.toLocaleString()}</td>
              <td className="px-6 py-5 text-red-500">${hospitalLateTotal.toLocaleString()}</td>
              <td className="px-6 py-5 text-green-600">${hospitalSavings.toLocaleString()}</td>
              {/* <td className="px-6 py-5 text-green-600">{hospitalSavingsPercentage}%</td> */}
            </tr>
          </tbody>
        </table>
      </div>

      {/* ROI Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gradient-to-br from-green-50 to-green-100 border border-green-200 rounded-2xl p-6 hover:scale-105 transition-transform duration-200">
          <div className="flex items-center space-x-3 mb-3">
            <DollarSign className="w-8 h-8 text-green-600" />
            <h3 className="text-xl font-bold text-slate-800">Total Savings</h3>
          </div>
          <p className="text-3xl font-bold text-green-600">${hospitalSavings.toLocaleString()}</p>
          <p className="text-slate-500 mt-2">Hospital-wide with early intervention</p>
        </div>

        <div className="bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-2xl p-6 hover:scale-105 transition-transform duration-200">
          <div className="flex items-center space-x-3 mb-3">
            <TrendingUp className="w-8 h-8 text-blue-600" />
            <h3 className="text-xl font-bold text-slate-800">Cost Reduction</h3>
          </div>
          <p className="text-3xl font-bold text-blue-600">{hospitalSavingsPercentage}%</p>
          <p className="text-slate-500 mt-2">Reduction in treatment costs</p>
        </div>

        <div className="bg-gradient-to-br from-green-100 to-blue-100 border border-slate-200 rounded-2xl p-6 hover:scale-105 transition-transform duration-200">
          <div className="flex items-center space-x-3 mb-3">
            <AlertTriangle className="w-8 h-8 text-green-700" />
            <h3 className="text-xl font-bold text-slate-800">Risk Reduction</h3>
          </div>
          <p className="text-3xl font-bold text-green-700">75%</p>
          <p className="text-slate-500 mt-2">Lower complication rates</p>
        </div>

        <div className="bg-gradient-to-br from-teal-50 to-teal-100 border border-teal-200 rounded-2xl p-6 hover:scale-105 transition-transform duration-200">
          <div className="flex items-center space-x-3 mb-3">
            <DollarSign className="w-8 h-8 text-teal-600" />
            <h3 className="text-xl font-bold text-slate-800">Avg. Savings</h3>
          </div>
          <p className="text-3xl font-bold text-teal-600">
            ${aggregatedPatients.length > 0 ? Math.round(hospitalSavings / aggregatedPatients.length).toLocaleString() : 0}
          </p>
          <p className="text-slate-500 mt-2">Per patient with early intervention</p>
        </div>
      </div>
    </div>
  );
}