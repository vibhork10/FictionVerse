import React, { useState, useEffect } from 'react';
import styles from './StoryWriting.module.css';
import DropdownMenu from '../../components/DropdownMenu/DropdownMenu';
import TextField from '../../components/TextField/TextField';
import Button from '../../components/Button/Button';
import axios from 'axios'; // Import Axios
import LinearProgress from '@mui/material/LinearProgress';
import { styled } from '@mui/system';

const ProgressBarContainer = styled('div')({
  width: '100%',
  marginTop: 10,
});


function StoryWriting({setGeneratedStory, switchToImageGenerationTab, story, setStory, genre, setGenre }) {
  // const [genre, setGenre] = useState('');
  // const [story, setStory] = useState('');
  const [rows, setRows] = useState(10);
  const [loading, setLoading] = useState(false);


  const handleSubmit = async () => {
    setLoading(true);
    const data = {
      "genre": genre,
      "user_input": story.toString() 
    };

    
    const response = await axios.post('http://127.0.0.1:8000/generate_story', data);
    setStory(response.data.story);
    
    setLoading(false);
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
            { value: 'thriller', label: 'Thriller'},
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
          <Button text="Generate" onClick={handleSubmit}/>
        </div>
        <div className={styles.buttonRight}>
          <Button text="Clear" color="--error-color" onClick={handleClearClick} />
          <Button text="Submit" onClick={handleSubmitClick} />
        </div>
      </div>
      {loading && (
      <ProgressBarContainer>
        <LinearProgress />
      </ProgressBarContainer>
    )}
    </div>
  );
}

export default StoryWriting;
