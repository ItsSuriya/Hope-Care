import React from 'react'
import AboutSection from '../components/About'
import FeaturesGrid from '../components/Features'
import { HowItWorks } from '../components/HowItWorks'
import { ParallaxFooter } from '../components/Footer'
import DemoOne from '@/components/Hero1'

const Home = () => {
  return (
    <div>
      <DemoOne />
      <AboutSection />
      <FeaturesGrid />
      <HowItWorks />
      {/* <DevelopersSection /> */}
      <ParallaxFooter />
      {/* <div className='min-h-screen'></div> */}
    </div>
  )
}

export default Home