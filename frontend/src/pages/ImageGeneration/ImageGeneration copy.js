import React, { useState } from 'react';
import styles from './ImageGeneration.module.css';
import TextField from '../../components/TextField/TextField';
import Button from '../../components/Button/Button';
import CustomDropdown from '../../components/DropdownMenu/CustomDropdown';
import axios from 'axios';
import CircularProgress from '@mui/material/CircularProgress';
import image1 from '../../Assets/midjourney.png';
import image2 from '../../Assets/anything-v4.0.png';
import image3 from '../../Assets/andreasrocha_artstyle.png';
import image4 from '../../Assets/charliebo_artstyle.png';
import image5 from '../../Assets/holliemengert_artstyle.png';
import image6 from '../../Assets/jamesdaly_artstyle.png';
import image7 from '../../Assets/marioalberti_artstyle.png';
import image8 from '../../Assets/pepelarraz_artstyle.png';
import image9 from '../../Assets/default_style.png';
import image10 from '../../Assets/comicstyle_story.png';

function ImageGeneration({generatedStory, currentLine, setCurrentLine, currentCount, setCurrentCount, generatedImageUrl, setGeneratedImageUrl, choice, setChoice, prompt, setPrompt, nwstyle,setstyle}) {
  const [loading, setLoading] = useState(false);
 
  const handleNext = async () => {
    setLoading(true);
    const data = {
      "input_text": generatedStory,
      "count": currentCount
    };

    const response = await axios.post('http://127.0.0.1:8000/next_line', data);
    setCurrentLine(response.data.line);
    setCurrentCount(response.data.count);
    setLoading(false);
  };

  const handlePrev = async () => {
    setLoading(true);
    const data = {
      "input_text": generatedStory,
      "count": currentCount - 1
    };

    const response = await axios.post('http://127.0.0.1:8000/prev_line', data);
    setCurrentLine(response.data.line);
    setCurrentCount(response.data.count + 1);
    setLoading(false);
  };



  // const handleStoryChange = event => {
  //   setStory(event.target.value);
  // };

  const handlePromptChange = event => {
    setPrompt(event.target.value);
  };

  const handleChoiceChange = (option) => {
    setChoice(option);
  };
  const handleStyleChange = (option) => {
    setstyle(option);
  };
  const handleGenerateClick = async () => {
    setLoading(true);
    const data = {
      "prompt": prompt,
      "line_box": currentCount,
      "org_text": currentLine,
      "style": choice.value,
      "display":nwstyle.value
    };
  
    const response = await axios.post('http://127.0.0.1:8000/load_sd', data);
    if (response.data.image === 'done') {
      const cacheBuster = new Date().getTime();
      const imageUrl = `http://localhost:8000/images/${currentCount}.png?${cacheBuster}`;
      setGeneratedImageUrl(imageUrl); // Uncomment this line
    } else {
      console.error('Image generation failed');
    }
    setLoading(false);
  };

  const handleClearClick = () => {
    setPrompt('');
    setChoice('');
  };

  const handleDownloadClick = async () => {
    try {
      const response = await axios.get('http://127.0.0.1:8000/download_pdf', {
        responseType: 'blob',
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'generated_images.pdf');
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
    } catch (error) {
      console.error('Error downloading the PDF', error);
    }
  };
  console.log(loading);
  return (
    <div className={styles.imageGeneration}>
      <div className={styles.leftContainer}>
        <TextField
          placeholder="Story from Story Writing page..."
          value={generatedStory}
          multiline
          rows={20}
        />
      </div>
      <div className={styles.rightContainer}>
        <div className={styles.finalStory}>
          <TextField
            placeholder="Original Line"
            value={currentLine}
            multiline
            rows={1}
          />
        </div>
        <div className={styles.navigation}>
          <Button text="Previous" onClick={handlePrev} />
          <TextField
            placeholder="Line number..."
            type="number"
            value={currentCount}
          />
          <Button text="Next" onClick={handleNext}/>
        </div>
        <div className={styles.prompt}>
          <TextField
            placeholder="Enter prompt..."
            value={prompt}
            onChange={handlePromptChange}
          />
        </div>
        <div className={styles.seedAndRegenerate}>
          <div className={styles.regenerate}>
            <CustomDropdown
              options={[
                { value: 'ogkalu/Comic-Diffusion-andreasrocha_artstyle', label: 'Comic style-1', icon: image3 },
                { value: 'ogkalu/Comic-Diffusion-charliebo_artstyle', label: 'Comic style-2', icon: image4 },
                { value: 'ogkalu/Comic-Diffusion-holliemengert_artstyle', label: 'Comic style-3', icon: image5 },
                { value: 'ogkalu/Comic-Diffusion-jamesdaly_artstyle', label: 'Comic style-4', icon: image6 },
                { value: 'ogkalu/Comic-Diffusion-marioalberti_artstyle', label: 'Comic style-5', icon: image7 },
                { value: 'ogkalu/Comic-Diffusion-pepelarraz_artstyle', label: 'Comic style-6', icon: image8 },
                { value: 'prompthero/openjourney-v4', label: 'Ultra-artistic', icon: image1 },
                { value: 'andite/anything-v4.0', label: 'Anime Style-1', icon: image2 },
              ]}
              value={choice}
              onChange={handleChoiceChange}
            />
            <CustomDropdown
              options={[
                { value: 'Comic-Style', label: 'Comic style-1', icon: image10},
                { value: 'default-style', label: 'Comic style-2', icon: image9},
              ]}
              value={nwstyle}
              onChange={handleStyleChange}
            />
          </div>
        </div>
        <div className={styles.buttons}>
          <Button text="Generate" onClick={handleGenerateClick} disabled={loading}>
            {loading && (
              <CircularProgress size={24} style={{ marginLeft: 10, color: 'white' }} />
            )}
          </Button>
          <Button text="Clear" color="--error-color" onClick={handleClearClick} />
        </div>
        <div className={styles.imageContainer}>
          {generatedImageUrl && <img src={generatedImageUrl} alt="Generated image" />}
        </div>
        <div className={styles.download}>
          <Button text="Download PDF" onClick={handleDownloadClick} />
        </div>
      </div>
    </div>
  );
}

export default ImageGeneration;
