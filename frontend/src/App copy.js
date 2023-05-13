import React from 'react';
import './styles/global.css';
import StoryWriting from './pages/StoryWriting/StoryWriting';
import ImageGeneration from './pages/ImageGeneration/ImageGeneration'; // add this line


function App() {
  return (
    <div className="App">
      {/* <StoryWriting /> */}
      <ImageGeneration /> {/* change this line */}
    </div>
  );
}

export default App;
