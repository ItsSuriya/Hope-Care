"use client";

import React from "react";

export function ParallaxFooter() {
  return (
    <footer className="relative h-[50vh] md:h-[60vh] w-full overflow-hidden">
      {/* Background Parallax Image */}
      <div
        className="absolute inset-0 bg-fixed bg-bottom bg-cover"
        // style={{
        //   backgroundImage: "url('/Footer.jpg')",
        // }}
      ></div>

      {/* Overlay with healthcare gradient */}
      <div className="absolute inset-0 bg-gradient-to-t from-emerald-500/30 to-transparent"></div>

      {/* Content */}
      <div className="relative z-10 flex flex-col items-center justify-center h-full text-center text-slate-900 px-6">
        <h2 className="text-3xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-[#6B4EFF] via-[#3B82F6] to-emerald-600 bg-clip-text text-transparent">
          Together for a Healthier Future
        </h2>

        <p className="text-slate-700 max-w-xl">
          Empowering healthcare through innovation, prevention, and protection.
        </p>

        <div className="mt-6 flex space-x-6">
          <a
            href="#"
            className="text-slate-600 hover:text-[#6B4EFF] transition-colors font-medium"
          >
            About Us
          </a>

        </div>

        <p className="mt-8 text-sm text-slate-500">
          © {new Date().getFullYear()} HopeCare. All Rights Reserved.
        </p>
      </div>
    </footer>
  );
}