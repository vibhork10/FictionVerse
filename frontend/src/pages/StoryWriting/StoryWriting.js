import React, { useState, useEffect } from 'react';
import styles from './StoryWriting.module.css';
import DropdownMenu from '../../components/DropdownMenu/DropdownMenu';
import TextField from '../../components/TextField/TextField';
import Button from '../../components/Button/Button';
import axios from 'axios'; // Import Axios
import LinearProgress from '@mui/material/LinearProgress';
import CircularProgress from '@mui/material/CircularProgress';
import { styled } from '@mui/system';

const ProgressBarContainer = styled('div')({
  width: '100%',
  marginTop: 10,
});


function StoryWriting({setGeneratedStory, switchToImageGenerationTab, story, setStory, genre, setGenre, errorMessage, setErrorMessage, storyloading, setstoryLoading}) {
  // const [genre, setGenre] = useState('');
  // const [story, setStory] = useState('');
  // const [errorMessage, setErrorMessage] = useState(null);
  const [rows, setRows] = useState(10);
  // const [loading, setLoading] = useState(false);


  const handleSubmit = async () => {
    setstoryLoading(true);
    setErrorMessage(null);
    const data = {
      "genre": genre,
      "user_input": story.toString() 
    };

    
    try {
      const response = await axios.post('http://54.157.42.127:8000/generate_story', data);
      console.log(response.data.story)
      if (response.data.story) {
        setStory(response.data.story);
        setstoryLoading(true);
      } else {
        // set error message for negative response
        setErrorMessage('Something went wrong. Please press the Generate Button again or check whether you have selected any genre or have given any input text or not.');
        setstoryLoading(false);
      }
    } catch (error) {
      // set error message for error response
      setErrorMessage('Something went wrong. Please press the Generate Button again or check whether you have selected any genre or have given any input text or not.');
      setstoryLoading(false);
    } finally {
      setstoryLoading(false);
    }
  };

  useEffect(() => {
    const calculateRows = () => {
      const screenHeight = window.innerHeight;
      const textAreaHeight = screenHeight * 0.7;
      const rowHeight = 10; // Estimated height per row, adjust as needed
      const numRows = Math.floor(textAreaHeight / rowHeight);
      setRows(numRows);
    };

    calculateRows();
    window.addEventListener('resize', calculateRows);

    return () => {
      window.removeEventListener('resize', calculateRows);
    };
  }, []);

  const handleGenreChange = event => {
    setGenre(event.target.value);
  };

  const handleStoryChange = event => {
    setStory(event.target.value);
  };

  const handleClearClick = () => {
    setStory('');
  };

  const handleSubmitClick = () => {
    // submit the story
    setGeneratedStory(story);
    switchToImageGenerationTab();
    console.log("Redirecting to the Image Generation Page!");
  };

  return (
    <div className={styles.storyWriting}>
      <div className={styles.dropdownContainer}>
        <DropdownMenu
          options={[
            { value: '', label: 'Select a genre' },
            { value: 'horror', label: 'Horror' },
            { value: 'comedy', label: 'Comedy' },
            { value: 'mystery', label: 'Mystery' },
            { value: 'historical_fiction', label: 'Historical Fiction'},
            { value: 'adventure', label: 'Adventure'},
            { value: 'science_fiction', label: 'Sci-fi'},
            { value: 'fantasy', label: 'Fantasy'}
          ]}
          value={genre}
          onChange={handleGenreChange}
        />
      </div>
      <div className={styles.textAreaContainer}>
        <TextField
          placeholder="Start typing your story..."
          value={story}
          onChange={handleStoryChange}
          multiline
          rows={rows}
        />
      </div>
      <div className={styles.buttonContainer}>
        <div className={styles.buttonLeft}>
          <Button text={storyloading ? "Generating..." : "Generate"} onClick={handleSubmit} >
            {storyloading && <CircularProgress size={24} color="inherit" />}
          </Button>
        </div>
        <div className={styles.buttonRight}>
          <Button text="Clear" color="--error-color" onClick={handleClearClick} />
          <Button text="Submit" onClick={handleSubmitClick} />
        </div>
      </div>
      {errorMessage && (
        <div className={styles.errorContainer}>
          <p>{errorMessage}</p>
        </div>
      )}
    </div>
  );
}

export default StoryWriting;
