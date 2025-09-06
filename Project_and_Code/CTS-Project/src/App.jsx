import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./Pages/Home";
import Analysis from "./Pages/Analysis";
import Dashboard from "./Pages/Dashboard";

function App() {
  return (
    <Router>
      {/* NavBar is outside Routes → visible on all pages */}
      <Navbar />

      <div className="pt-20"> {/* optional padding to avoid overlap with fixed navbar */}
        <Routes>
          <Route path="/" element={<Home />} />
          {/* <Route path="/about-us" element={<AboutUs />} /> */}
          <Route path="/analysis" element={<Analysis />} />
          {/* <Route path="/result" element={<Result />} /> */}
          <Route path="/dashboard" element={<Dashboard />}  />
        </Routes>
      </div>
    </Router>
  );
}

export default App;