import React from 'react';

export default function PatientTable({ patients }) {
  const getRiskTierColor = (tier) => {
    if (!tier) return "bg-gray-200 text-gray-600 border-gray-300";
    if (tier.includes("High")) return "bg-red-100 text-red-600 border-red-300";
    if (tier.includes("Moderate")) return "bg-yellow-100 text-yellow-600 border-yellow-300";
    if (tier.includes("Low")) return "bg-green-100 text-green-600 border-green-300";
    return "bg-blue-100 text-blue-600 border-blue-300";
  };

  if (!patients || patients.length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-3xl p-8 shadow-lg h-full flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-500 text-sm">
            No patient data to display. Upload a CSV file to see the predictions.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-3xl p-8 shadow-lg h-full flex flex-col">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-500 to-blue-500">
          Prediction Results
        </h2>
      </div>

      <div className="overflow-auto flex-grow rounded-2xl border border-gray-200">
        <table className="w-full text-sm min-w-max">
          <thead>
            <tr className="bg-gradient-to-r from-green-500 to-blue-500 sticky top-0 z-10">
              {[
                "Patient ID",
                "Age",
                "Location",
                "Income",
                "Employment",
                "Hospital Visits (2yr)",
                "Primary Condition",
                "Risk Score",
                "Risk Tier",
                "Key Risk Factors",
              ].map((header, idx) => (
                <th
                  key={idx}
                  className="px-4 py-3 text-left font-bold text-white text-sm uppercase tracking-wide"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {patients.map((pred, index) => (
              <tr
                key={index}
                className="border-b border-gray-200 hover:bg-gradient-to-r hover:from-green-50 hover:to-blue-50 transition-colors duration-200"
              >
                <td className="px-4 py-3 text-gray-800 font-medium">{pred.patientId}</td>
                <td className="px-4 py-3 text-gray-800">{pred.age}</td>
                <td className="px-4 py-3 text-gray-800">{pred.location}</td>
                <td className="px-4 py-3 text-gray-800">{pred.income}</td>
                <td className="px-4 py-3 text-gray-800">{pred.employment}</td>
                <td className="px-4 py-3 text-gray-800">{pred.hospital_visit}</td>
                <td className="px-4 py-3 text-gray-800">{pred.primary_condition}</td>
                <td className="px-4 py-3 text-gray-900 font-semibold">{pred.risk_score}</td>
                <td className="px-4 py-3">
                  <span
                    className={`px-3 py-1.5 rounded-lg text-xs font-bold border ${getRiskTierColor(
                      pred.risk_tier
                    )}`}
                  >
                    {pred.risk_tier}
                  </span>
                </td>
                <td className="px-4 py-3 text-gray-600 text-xs">
                  {(pred.key_risk_factors || []).join(', ')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}