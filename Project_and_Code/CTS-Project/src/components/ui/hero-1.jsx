import { ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion"; // <- Import Framer Motion

export function Hero({
  title,
  subtitle,
  ctaLabel = "Explore Now",
}) {
  const navigate = useNavigate();
  const handleCTAClick = () => navigate("/analysis");

  // Animation variants for fade-in with upward motion
  const fadeInUp = {
    hidden: { opacity: 0, y: 20 },
    visible: (custom = 0) => ({
      opacity: 1,
      y: 0,
      transition: { delay: custom * 0.2, duration: 0.8, ease: "easeInOut" },
    }),
  };

  return (
    <section
      id="hero"
      className="relative mx-auto w-full pt-40 px-6 text-center md:px-8 
      min-h-[calc(100vh-40px)] overflow-hidden 
      bg-white 
      dark:bg-[linear-gradient(to_bottom,#0f172a,#0891b2_30%,#06b6d4_78%,#14b8a6_99%)] 
      rounded-b-xl"
    >
      {/* Grid BG */}
      <div className="absolute -z-10 inset-0 opacity-60 h-[600px] w-full 
        bg-[linear-gradient(to_right,#0c9488_1px,transparent_1px),linear-gradient(to_bottom,#0c9488_1px,transparent_1px)] 
        dark:bg-[linear-gradient(to_right,#0891b2_1px,transparent_1px),linear-gradient(to_bottom,#0891b2_1px,transparent_1px)]
        bg-[size:6rem_5rem] 
        [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)]"
      />

      {/* Radial Accent */}
      <div className="absolute left-1/2 top-[calc(100%-90px)] lg:top-[calc(100%-150px)] 
        h-[500px] w-[700px] md:h-[500px] md:w-[1100px] lg:h-[750px] lg:w-[140%] 
        -translate-x-1/2 rounded-[100%] border-[#14b8a6] 
        bg-[radial-gradient(closest-side,#06f1ff_70%,#0d9488_100%)] 
        dark:bg-[radial-gradient(closest-side,#0ff_70%,#0891b2_100%)]"
      />

      {/* Title */}
      <motion.h1
        className="text-balance bg-gradient-to-br from-blue-700 to-emerald-500 
          bg-clip-text py-6 text-5xl font-semibold leading-none tracking-tighter text-transparent sm:text-6xl md:text-7xl lg:text-8xl"
        initial="hidden"
        animate="visible"
        custom={1}
        variants={fadeInUp}
      >
        {title}
      </motion.h1>

      {/* Subtitle */}
      <motion.p
        className="text-lg tracking-tight text-cyan-700 md:text-xl mt-4"
        initial="hidden"
        animate="visible"
        custom={2}
        variants={fadeInUp}
      >
        {subtitle}
      </motion.p>

      {/* CTA */}
      {ctaLabel && (
        <motion.div
          className="flex justify-center mt-6"
          initial="hidden"
          animate="visible"
          custom={3}
          variants={fadeInUp}
        >
          <Button
            variant="ghost"
            onClick={handleCTAClick}
            className="group rounded-[1.15rem] px-8 py-6 text-lg font-semibold backdrop-blur-md 
              bg-gradient-to-r from-blue-600 to-emerald-500
              text-white transition-all duration-300 border border-blue-200 hover:shadow-md hover:scale-110"
          >
            <span className="transition-colors duration-300 group-hover:text-white">
              Get Started
            </span>
            <span
              className="ml-3 opacity-70 group-hover:text-white group-hover:translate-x-1.5 
              transition-all duration-300"
            >
              →
            </span>
          </Button>
        </motion.div>
      )}
    </section>
  );
}