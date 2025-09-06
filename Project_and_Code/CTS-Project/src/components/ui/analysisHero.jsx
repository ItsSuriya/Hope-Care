"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/analysisbutton";

function FloatingPaths({ position }) {
  const paths = Array.from({ length: 36 }, (_, i) => ({
    id: i,
    d: `M-${380 - i * 5 * position} -${189 + i * 6}C-${380 - i * 5 * position} -${
      189 + i * 6
    } -${312 - i * 5 * position} ${216 - i * 6} ${152 - i * 5 * position} ${
      343 - i * 6
    }C${616 - i * 5 * position} ${470 - i * 6} ${684 - i * 5 * position} ${
      875 - i * 6
    } ${684 - i * 5 * position} ${875 - i * 6}`,
    color: `rgba(37, 99, 235, ${0.05 + i * 0.02})`, // blue-600 light paths
    width: 0.5 + i * 0.03,
  }));

  return (
    <div className="absolute inset-0 pointer-events-none">
      <svg className="w-full h-full text-emerald-600" viewBox="0 0 696 316" fill="none">
        <title>Background Paths</title>
        {paths.map((path) => (
          <motion.path
            key={path.id}
            d={path.d}
            stroke="currentColor"
            strokeWidth={path.width}
            strokeOpacity={0.05 + path.id * 0.02}
            initial={{ pathLength: 0.3, opacity: 0.5 }}
            animate={{
              pathLength: 1,
              opacity: [0.2, 0.5, 0.2],
              pathOffset: [0, 1, 0],
            }}
            transition={{
              duration: 20 + Math.random() * 10,
              repeat: Number.POSITIVE_INFINITY,
              ease: "linear",
            }}
          />
        ))}
      </svg>
    </div>
  );
}

// 🔹 Custom smooth scroll
function smoothScrollBy(distance, duration) {
  const start = window.scrollY;
  const startTime = performance.now();

  const step = (currentTime) => {
    const progress = Math.min((currentTime - startTime) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    window.scrollTo(0, start + distance * ease);
    if (progress < 1) requestAnimationFrame(step);
  };

  requestAnimationFrame(step);
}

export function BackgroundPaths({ title = "Background Paths" }) {
  const words = title.split(" ");

  return (
    <div className="relative h-[95vh] w-full flex items-center justify-center overflow-hidden bg-white">
      {/* Floating background paths */}
      <div className="absolute inset-0">
        <FloatingPaths position={1} />
        <FloatingPaths position={-1} />
      </div>

      {/* Center Content */}
      <div className="relative z-10 container mx-auto px-4 md:px-6 text-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 2 }}
          className="max-w-4xl mx-auto"
        >
         <h1 className="text-3xl sm:text-5xl md:text-7xl font-bold mt-auto mb-8 tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-teal-400 to-cyan-500">
  {words.map((word, wordIndex) => (
    <span key={wordIndex} className="inline-block mr-4 last:mr-0">
      {word.split("").map((letter, letterIndex) => (
        <motion.span
          key={`${wordIndex}-${letterIndex}`}
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{
            delay: wordIndex * 0.1 + letterIndex * 0.03,
            type: "spring",
            stiffness: 150,
            damping: 25,
          }}
          className="inline-block"
        >
          {letter}
        </motion.span>
      ))}
    </span>
  ))}
</h1>

          {/* Button */}
          <div
            className="inline-block group relative bg-gradient-to-b from-blue-100 to-green-100 
            p-px rounded-2xl backdrop-blur-lg overflow-hidden shadow-md hover:shadow-xl transition-shadow duration-300"
          >
            <Button
              variant="ghost"
              onClick={() => smoothScrollBy(600, 2000)}
              className="rounded-[1.15rem] px-8 py-6 text-lg font-semibold backdrop-blur-md 
                bg-gradient-to-r from-blue-600 to-emerald-500 
                text-white transition-all duration-300 
                group-hover:-translate-y-0.5 border border-blue-200 hover:shadow-md"
            >
              <span className=" group-hover:text-white transition-opacity">
                Get Started
              </span>
              <span
                className="ml-3  group-hover:text-white group-hover:translate-x-1.5 
                transition-all duration-300"
              >
                →
              </span>
            </Button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}