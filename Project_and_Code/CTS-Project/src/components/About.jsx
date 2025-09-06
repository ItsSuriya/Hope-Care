import React from "react"
import { motion } from "framer-motion"
import { useNavigate } from "react-router-dom"

const AboutSection = () => {
  const navigate = useNavigate()

  return (
    <section className="relative w-full bg-white py-16 px-6 md:px-12 lg:px-20">
      <div className="max-w-7xl mx-auto grid md:grid-cols-2 gap-12 items-center">
        
        {/* Text Section */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          whileInView={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="space-y-6"
        >
          <h2 className="text-3xl md:text-4xl font-extrabold text-slate-800">
            About{" "}
            <span className="font-island text-6xl bg-gradient-to-r from-blue-600 to-emerald-500 bg-clip-text text-transparent">
              Hope Care
            </span>
          </h2>

          <p className="text-slate-700 text-lg leading-relaxed">
            Healthcare today is often <span className="font-semibold text-blue-600">reactive</span>. 
            Care teams wait for a crisis to occur before they step in. 
            Hope Care changes that story — by giving clinicians a proactive 
            co-pilot that surfaces hidden risks <em>before</em> they become emergencies.
          </p>

          <p className="text-slate-600 leading-relaxed">
            Imagine Priya, a care manager, responsible for 500 patients. 
            Without Hope Care, she only sees patients ranked by basic conditions. 
            With our AI, subtle signals — like ER visits or late refills — 
            are analyzed overnight. 
          </p>

          <p className="text-slate-600 leading-relaxed">
            Suddenly, Mr. Sharma, who looked stable on paper, rises to the top of 
            her priority list. Hope Care doesn’t just say <em>“high risk”</em> — it explains 
            <span className="font-medium text-emerald-600"> why</span>, and even suggests the 
            <span className="font-medium text-purple-600"> right next action</span>.  
          </p>

          {/* Button to navigate */}
          <button
            onClick={() => navigate("/analysis")}
            className="mt-4 px-6 py-3 rounded-full bg-gradient-to-r from-blue-600 to-emerald-500 
                       text-white font-semibold shadow-lg hover:shadow-lg hover:scale-105 
                       transition-all duration-300"
          >
            Explore the Platform
          </button>
        </motion.div>

        {/* Image Section */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          whileInView={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="flex justify-center"
        >
          <div className="relative w-full max-w-md">
            <img
              src="/home.jpg"
              alt="Healthcare AI dashboard"
              className="rounded-2xl shadow-xl"
            />
            {/* Soft Gradient Overlay Glow */}
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-tr from-blue-200/40 to-emerald-200/40 blur-2xl -z-10" />
          </div>
        </motion.div>
      </div>
    </section>
  )
}

export default AboutSection