import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

// --- Build Patient Info table ---
function buildPatientInfoTable(patientData) {
  const head = [["Patient ID", "Age", "Primary Condition"]];
  const body = patientData.map((p) => [
    p.DESYNPUF_ID || p.patientId || "N/A", // Patient ID
    p.age !== undefined && p.age !== null ? p.age : "N/A", // ✅ Age check
    p.primary_condition || "N/A", // Primary condition
  ]);
  return { head, body };
}

// --- Build Risk & Outcomes table ---
function buildOutcomesTable(patientData) {
  const head = [["Condition", "Risk Score", "Risk Tier", "Key Risk Factors"]];
  let body = [];

  patientData.forEach((p) => {
    if (Array.isArray(p.predictedOutcomes)) {
      p.predictedOutcomes.forEach((outcome) => {
        body.push([
          outcome.condition || "",
          outcome.riskScore || "",
          outcome.riskTier || "",
          (outcome.keyRiskFactors || []).join(", "),
        ]);
      });
    }
  });

  return { head, body };
}

// --- Build ROI table with Primary Condition ---
function buildRoiTable(roiData) {
  const head = [
    ["Patient ID", "Age", "Primary Condition", "Proactive Cost", "Reactive Cost", "Potential Savings"],
  ];
  let body = [];

  roiData.forEach((r) => {
    if (Array.isArray(r.predictedCosts)) {
      r.predictedCosts.forEach((cost) => {
        body.push([
          r.patientId || r.DESYNPUF_ID || "N/A",
          r.age_used_for_prediction !== undefined && r.age_used_for_prediction !== null
            ? r.age_used_for_prediction
            : "N/A", // ✅ Age check
          cost.condition || r.primary_condition || "N/A",
          cost.predicted_proactive_cost || "N/A",
          cost.predicted_reactive_cost || "N/A",
          cost.potential_savings || "N/A",
        ]);
      });
    }
  });

  return { head, body };
}

export function generatePDF(patientData, roiData) {
  const doc = new jsPDF({ orientation: "landscape", unit: "pt", format: "a4" });

  // --- Title ---
  doc.setFontSize(20);
  doc.text("📊 Analysis Report", 40, 40);
  doc.setFontSize(11);
  doc.text(`Generated on: ${new Date().toLocaleString()}`, 40, 60);

  let yPos = 90;

  // --- Patient Info Table ---
  if (patientData.length > 0) {
    const { head, body } = buildPatientInfoTable(patientData);

    doc.setFontSize(14);
    doc.text("Patient Information", 40, yPos);

    autoTable(doc, {
      startY: yPos + 10,
      head,
      body,
      theme: "striped",
      styles: { fontSize: 9, cellPadding: 4, halign: "center", valign: "middle" },
      headStyles: { fillColor: [46, 125, 50], textColor: 255, halign: "center" },
      bodyStyles: { halign: "center" },
      alternateRowStyles: { fillColor: [245, 255, 245] },
    });

    yPos = doc.lastAutoTable.finalY + 30;
  }

  // --- Risk & Outcomes Table ---
  const outcomesTable = buildOutcomesTable(patientData);
  if (outcomesTable.body.length > 0) {
    doc.setFontSize(14);
    doc.text("Risk & Outcomes", 40, yPos);

    autoTable(doc, {
      startY: yPos + 10,
      head: outcomesTable.head,
      body: outcomesTable.body,
      theme: "striped",
      styles: { fontSize: 9, cellPadding: 4, halign: "center", valign: "middle" },
      headStyles: { fillColor: [37, 99, 235], textColor: 255, halign: "center" },
      bodyStyles: { halign: "center" },
      alternateRowStyles: { fillColor: [240, 248, 255] },
    });

    yPos = doc.lastAutoTable.finalY + 30;
  }

  // --- ROI Data Table ---
  const roiTable = buildRoiTable(roiData);
  if (roiTable.body.length > 0) {
    doc.setFontSize(14);
    doc.text("ROI Data", 40, yPos);

    autoTable(doc, {
      startY: yPos + 10,
      head: roiTable.head,
      body: roiTable.body,
      theme: "grid",
      styles: { fontSize: 9, cellPadding: 4, halign: "center", valign: "middle" },
      headStyles: { fillColor: [200, 80, 0], textColor: 255, halign: "center" },
      bodyStyles: { halign: "center" },
      alternateRowStyles: { fillColor: [255, 245, 230] },
    });
  }

  doc.save("analysis_report.pdf");
}