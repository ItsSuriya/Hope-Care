import React, { useState, useEffect } from "react"
import { FiMenu, FiX } from "react-icons/fi"
import { Link, useLocation } from "react-router-dom"

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false)
  const location = useLocation()
  const [active, setActive] = useState("Home")

  const navItems = [
    { name: "Home", url: "/" },
    { name: "Analysis", url: "/analysis" },
    { name: "Dashboard", url: "/dashboard" },
  ]

  // Sync active state with current URL
  useEffect(() => {
    const currentItem = navItems.find(item => item.url === location.pathname)
    if (currentItem) setActive(currentItem.name)
  }, [location.pathname])

  return (
    <nav className="w-full fixed top-0 left-0 z-50 bg-white/20 backdrop-blur border-b border-white/30 shadow-lg">
      <div className="w-full px-6 py-4 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" onClick={() => setActive("Home")}>
          <div className="text-7xl md:text-5xl font-island font-extrabold tracking-wide text-transparent bg-clip-text bg-gradient-to-l from-emerald-500 to-[#3B82F6] drop-shadow-sm">
            Hope Care
          </div>
        </Link>

        {/* Desktop Menu */}
        <div className="hidden md:flex space-x-6">
          {navItems.map((item) => (
            <Link
              key={item.name}
              to={item.url}
              onClick={() => setActive(item.name)}
              className={`px-5 py-2 rounded-full font-medium transition-all duration-300
                ${
                  active === item.name
                    ? "bg-gradient-to-l from-emerald-400/90 to-[#3B82F6]/90 text-white shadow-lg shadow-blue-500/30 scale-105"
                    : "text-gray-800 hover:text-emerald-500 hover:bg-white/30"
                }`}
            >
              {item.name}
            </Link>
          ))}
        </div>

        {/* Mobile Menu Button */}
        <button
          className="md:hidden text-3xl text-gray-800"
          onClick={() => setIsOpen(!isOpen)}
        >
          {isOpen ? <FiX /> : <FiMenu />}
        </button>
      </div>

      {/* Mobile Dropdown */}
      <div
        className={`md:hidden transition-all duration-500 overflow-hidden ${
          isOpen ? "max-h-96" : "max-h-0"
        }`}
      >
        <div className="backdrop-blur-2xl bg-white/30 border-t border-white/40 flex flex-col items-stretch space-y-4 py-6 px-6 rounded-b-2xl shadow-xl">
          {navItems.map((item) => (
            <Link
              key={item.name}
              to={item.url}
              onClick={() => {
                setActive(item.name)
                setIsOpen(false)
              }}
              className={`text-center px-5 py-2 rounded-full font-medium transition-all duration-300
                ${
                  active === item.name
                    ? "bg-gradient-to-r from-[#6B4EFF]/90 to-[#3B82F6]/90 text-white shadow-lg shadow-blue-500/30 scale-105"
                    : "text-gray-800 hover:text-[#6B4EFF] hover:bg-white/40"
                }`}
            >
              {item.name}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}

export default Navbar