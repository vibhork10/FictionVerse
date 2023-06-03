import React, { useState } from 'react';
import styles from './ImageGeneration.module.css';
import TextField from '../../components/TextField/TextField';
import Button from '../../components/Button/Button';
import CustomDropdown from '../../components/DropdownMenu/CustomDropdown';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import CircularProgress from '@mui/material/CircularProgress';
// import LinearProgress from '@mui/material/LinearProgress';

import comic1 from '../../Assets/style_1.png';
import comic2 from '../../Assets/style_2.png';
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
import image0 from '../../Assets/sd_art.png';

function ImageGeneration({generatedStory, currentLine, setCurrentLine, currentCount, setCurrentCount, generatedImageUrl, setGeneratedImageUrl, choice, setChoice, prompt, setPrompt, nwstyle,setstyle, nwuuid,setnwuuID,loading,setLoading, nwseed, setSeed}) {
 
  const handleNext = async () => {
    setLoading(true);
    const data = {
      "input_text": generatedStory,
      "count": currentCount
    };




    const response = await axios.post('http://127.0.0.1:8000/next_line', data);
    setCurrentLine(response.data.line);
    setPrompt(response.data.line);
    setCurrentCount(response.data.count);
    setLoading(false);
  };
  const openForm = () => {
    window.open("https://docs.google.com/forms/d/e/1FAIpQLScz6hDLB3hBbj0M7HCRQ6aj0Q3hg1JRi5TDIWjIST6Wo8uatA/viewform?usp=sf_link", "_blank");
  };
  const handlePrev = async () => {
    setLoading(true);
    const data = {
      "input_text": generatedStory,
      "count": currentCount - 1
    };

    const response = await axios.post('http://127.0.0.1:8000/prev_line', data);
    setCurrentLine(response.data.line);
    setPrompt(response.data.line);
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
      "display":nwstyle.value,
      "uuid": nwuuid, 
      "seed": nwseed
    };
  
    try {
      const response = await axios.post('http://127.0.0.1:8000/load_sd', data);
      if (response.data.image === 'done') {
        setSeed(response.data.seed)
        const cacheBuster = new Date().getTime();
        const imageUrl = `http://localhost:8000/${response.data.uuid}_images/${currentCount}.png?${cacheBuster}`;
        setGeneratedImageUrl(imageUrl); // Uncomment this line
      } else {
        const errorMessage = 'Image generation failed. Please press the generate button again.';
        alert(errorMessage); // pop-up alert
        // console.error(errorMessage);
      }
    } catch (error) {
        const errorMessage = 'Image generation failed. Please press the generate button again.';
        alert(errorMessage); // pop-up alert
      // console.error(errorMessage, error);
    } finally {
      setLoading(false);
    }
  };
  const handleClearClick = () => {
    setPrompt(''); // clear prompt
    setChoice('Select Styles'); // clear choice
    setstyle('Select display type');  // clear style
    setnwuuID(uuidv4()); // set new UUID
    setGeneratedImageUrl(''); // clear image
    setCurrentCount(0); // reset count
    setCurrentLine(''); // clear current line
  };

  const handleDownloadClick = async () => {
    try {
      const response = await axios.get(`http://127.0.0.1:8000/download_pdf?uuid=${nwuuid}`, {
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
      alert('Error downloading the PDF, you may not have any generated image. Please generate an image.')
      console.error('Error downloading the PDF', error);
    }
  };
  // console.log(loading);
  return (
    <div className={styles.imageGeneration}>
      <div className={styles.leftContainer}>
        <TextField
          placeholder="Story from Story Writing page..."
          value={generatedStory}
          multiline
          rows={20}
          readOnly={true}
        />
      </div>
      <div className={styles.rightContainer}>
        <div className={styles.finalStory}>
          <TextField
            placeholder="Original Line"
            value={currentLine}
            multiline
            rows={1}
            readOnly={true}
          />
        </div>
        <div className={styles.navigation}>
          <Button text="Previous" onClick={handlePrev} />
          <TextField
            placeholder="Line number..."
            type="number"
            value={currentCount}
            readOnly={true}
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
                { value: 'runwayml/stable-diffusion-v1-5', label: 'General art', icon: image0},
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
                { value: 'default-style', label: 'Default-style', icon: image9},
                { value: 'Style-1', label: 'Comic style-1', icon: comic2},
                { value: 'Style-2', label: 'Comic style-2', icon: image10},
              ]}
              value={nwstyle}
              onChange={handleStyleChange}
            />
          </div>
        </div>
        <div className={styles.buttons}>
          <Button 
            text={loading ? "Generating..........." : "Generate"} 
            onClick={handleGenerateClick} 
            disabled={loading}
          >
            {loading && <CircularProgress size={24} color="inherit" />}
          </Button>
          <Button text="Clear" color="--error-color" onClick={handleClearClick} />
        </div>
        <div className={styles.imageContainer}>
          {generatedImageUrl && <img src={generatedImageUrl} alt="Generated image" />}
        </div>
        <div className={styles.download}>
          <Button text="Download PDF" onClick={handleDownloadClick} />
          <div>
            <Button text="Give Feedback" onClick={openForm} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default ImageGeneration;
