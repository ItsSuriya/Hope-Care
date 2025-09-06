import React, { useState, useRef } from "react";
import { Upload, File, X, CheckCircle } from "lucide-react";
import Papa from "papaparse";
import toast, { Toaster } from "react-hot-toast";

export default function CSVUploadComponent({ onAnalysis }) {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const fileInputRef = useRef(null);

  // ✅ Required & optional fields
  const REQUIRED_FIELDS = [
    "DESYNPUF_ID",
    "Age",
    "chronic_condition_count",
    "SP_CHF",
    "SP_CHRNKIDN",
    "SP_COPD",
    "inpatient_visit_count_08_09",
    "outpatient_visit_count_08_09",
    "unique_drug_count_08_09",
    "acute_heart_failure_claim_count_08_09",
    "acute_kidney_injury_claim_count_08_09",
    "copd_exacerbation_claim_count_08_09",
  ];

  const OPTIONAL_FIELDS = ["location", "employee", "income", "phone_number"];

  const validateCSVHeaders = (file) => {
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        header: true,
        preview: 1,
        complete: (results) => {
          if (results.meta && results.meta.fields) {
            const headers = results.meta.fields;
            const missingRequired = REQUIRED_FIELDS.filter(
              (field) => !headers.includes(field)
            );

            if (missingRequired.length > 0) {
              toast.error(
                `CSV is missing required fields: ${missingRequired.join(", ")}`,
                { style: { borderRadius: "12px", background: "#fee2e2", color: "#991b1b" } }
              );
              reject(missingRequired);
            } else {
              toast.success("CSV validated successfully!", {
                style: { borderRadius: "12px", background: "#ecfdf5", color: "#065f46" },
              });
              resolve(true);
            }
          } else {
            toast.error("Invalid CSV file format!", {
              style: { borderRadius: "12px", background: "#fee2e2", color: "#991b1b" },
            });
            reject("Invalid format");
          }
        },
        error: (error) => {
          console.error("Error parsing CSV:", error);
          toast.error("Error reading the CSV file", {
            style: { borderRadius: "12px", background: "#fee2e2", color: "#991b1b" },
          });
          reject(error);
        },
      });
    });
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    const csvFile = droppedFiles.find(
      (file) => file.type === "text/csv" || file.name.endsWith(".csv")
    );

    if (csvFile) {
      setFile(csvFile);
      setIsSubmitted(false);
    }
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (
      selectedFile &&
      (selectedFile.type === "text/csv" || selectedFile.name.endsWith(".csv"))
    ) {
      setFile(selectedFile);
      setIsSubmitted(false);
    }
  };

  const handleRemoveFile = () => {
    setFile(null);
    setIsSubmitted(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    try {
      await validateCSVHeaders(file);
    } catch (error) {
      return; // ❌ stop submission if validation fails
    }

    setIsSubmitting(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        if (data) {
          onAnalysis(data);
          toast.success("File submitted successfully!", {
            style: { borderRadius: "12px", background: "#ecfdf5", color: "#065f46" },
          });
        }
        setIsSubmitted(true);
      } else {
        toast.error("Error submitting file", {
          style: { borderRadius: "12px", background: "#fee2e2", color: "#991b1b" },
        });
      }
    } catch (error) {
      console.error("Error:", error);
      toast.error("Something went wrong!", {
        style: { borderRadius: "12px", background: "#fee2e2", color: "#991b1b" },
      });
    } finally {
      setIsSubmitting(false);
      setTimeout(() => {
        setIsSubmitted(false);
        setFile(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }, 3000);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <div className="max-h-[80vh] bg-white flex items-center justify-center p-6">
      {/* ✅ Modern toast container */}
      <Toaster position="bottom-right" reverseOrder={false} />

      <div className="w-full max-w-md">
        <div className="space-y-6">
          {/* Upload Area */}
          <div
            className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 cursor-pointer group
              ${
                isDragging
                  ? "border-green-400 bg-green-50"
                  : file
                  ? "border-blue-400 bg-blue-50"
                  : "border-slate-300 hover:border-green-400 hover:bg-green-50/50"
              }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => !file && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              className="hidden"
            />

            {!file ? (
              <>
                <div
                  className={`mx-auto w-16 h-16 rounded-full flex items-center justify-center mb-4 transition-all duration-300
                  ${
                    isDragging
                      ? "bg-green-100"
                      : "bg-blue-100 group-hover:bg-green-100"
                  }`}
                >
                  <Upload
                    className={`w-8 h-8 transition-colors duration-300 
                    ${
                      isDragging
                        ? "text-green-500"
                        : "text-blue-500 group-hover:text-green-500"
                    }`}
                  />
                </div>
                <h3 className="text-xl font-semibold text-slate-800 mb-2">
                  {isDragging ? "Drop your CSV file here" : "Upload CSV File"}
                </h3>
                <p className="text-slate-500 text-sm">
                  Drag and drop your file here, or click to browse
                </p>
                <p className="text-slate-400 text-xs mt-2">
                  Only CSV files are supported
                </p>
              </>
            ) : (
              <div className="space-y-3">
                <div className="mx-auto w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
                  <File className="w-8 h-8 text-blue-500" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-800 truncate">
                    {file.name}
                  </h3>
                  <p className="text-slate-500 text-sm">
                    {formatFileSize(file.size)}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={handleRemoveFile}
                  className="inline-flex items-center gap-2 px-3 py-1.5 text-sm text-red-500 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors duration-200"
                >
                  <X className="w-4 h-4" />
                  Remove
                </button>
              </div>
            )}
          </div>

          {/* Submit Button */}
          <button
            onClick={handleSubmit}
            disabled={!file || isSubmitting}
            className={`w-full rounded-[1.15rem] px-8 py-4 text-lg font-semibold transition-all duration-300 border
              ${
                !file || isSubmitting
                  ? "bg-slate-100 text-slate-400 border-slate-200 cursor-not-allowed"
                  : isSubmitted
                  ? "bg-green-500 hover:bg-green-600 text-white border-green-500"
                  : "bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 text-white shadow-md hover:shadow-lg hover:-translate-y-0.5"
              }`}
          >
            {isSubmitting ? (
              <div className="flex items-center justify-center gap-3">
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                Processing...
              </div>
            ) : isSubmitted ? (
              <div className="flex items-center justify-center gap-3">
                <CheckCircle className="w-5 h-5" />
                Submitted Successfully!
              </div>
            ) : (
              "Submit CSV File"
            )}
          </button>
        </div>

        {/* Info Text */}
        <div className="mt-6 text-center">
          <p className="text-slate-500 text-sm">
            Upload your CSV file to process and analyze your data
          </p>
        </div>
      </div>
    </div>
  );
}