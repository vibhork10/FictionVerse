import React, { useState } from 'react';
import './styles/global.css';
import StoryWriting from './pages/StoryWriting/StoryWriting';
import ImageGeneration from './pages/ImageGeneration/ImageGeneration';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import { styled } from '@mui/system';
import { v4 as uuidv4 } from 'uuid';

const TabContainer = styled('div')({
  flexGrow: 1,
  backgroundColor: 'white',
});

function App() {
  const [nwseed, setSeed] = useState(12);
  const [nwuuid, setnwuuID] = useState(uuidv4());
  const [storyloading, setstoryLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  const [generatedStory, setGeneratedStory] = useState('');
  const [prompt, setPrompt] = useState('');
  const [story, setStory] = useState('');
  const [genre, setGenre] = useState('');
  const [currentLine, setCurrentLine] = useState('Empty line');
  const [currentCount, setCurrentCount] = useState(0);
  const [generatedImageUrl, setGeneratedImageUrl] = useState('');
  const [choice, setChoice] = useState({
    value: 'prompthero/openjourney-v4',
    label: 'Select Styles',
  });
  const [nwstyle, setstyle] = useState({
    value: 'default',
    label: 'Select display type',
  });
  const handleChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const switchToImageGenerationTab = () => {
    setTabValue(1);
  };

  return (
    <div className="App">
      <TabContainer>
        <Tabs value={tabValue} onChange={handleChange} centered>
          <Tab label="Story Writing" />
          <Tab label="Image Generation" />
        </Tabs>
        <div role="tabpanel" hidden={tabValue !== 0}>
          {tabValue === 0 && (
            <StoryWriting
              setGeneratedStory={setGeneratedStory}
              switchToImageGenerationTab={switchToImageGenerationTab}
              story={story}
              setStory={setStory}
              genre={genre}
              setGenre={setGenre}
              errorMessage={errorMessage}
              setErrorMessage={setErrorMessage}
              storyloading={storyloading}
              setstoryLoading={setstoryLoading}
            />
          )}
        </div>
        <div role="tabpanel" hidden={tabValue !== 1}>
          {tabValue === 1 && 
          <ImageGeneration 
            generatedStory={generatedStory} 
            currentLine={currentLine}
            setCurrentLine={setCurrentLine}
            currentCount={currentCount}
            setCurrentCount={setCurrentCount}
            generatedImageUrl={generatedImageUrl}
            setGeneratedImageUrl={setGeneratedImageUrl}
            choice={choice}
            setChoice={setChoice}
            prompt={prompt}
            setPrompt={setPrompt}
            nwstyle={nwstyle}
            setstyle={setstyle}
            nwuuid={nwuuid}
            setnwuuID={setnwuuID}
            loading={loading}
            setLoading={setLoading}
            nwseed={nwseed}
            setSeed={setSeed}
          />}
        </div>
      </TabContainer>
    </div>
  );
}

export default App;
