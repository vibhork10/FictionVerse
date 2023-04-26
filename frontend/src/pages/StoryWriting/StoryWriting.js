import React, { useState, useEffect } from 'react';
import styles from './StoryWriting.module.css';
import DropdownMenu from '../../components/DropdownMenu/DropdownMenu';
import TextField from '../../components/TextField/TextField';
import Button from '../../components/Button/Button';

function StoryWriting() {
  const [genre, setGenre] = useState('');
  const [story, setStory] = useState('');
  const [rows, setRows] = useState(10);

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
            { value: 'thriller', label: 'Thriller' }
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
          <Button text="Generate" />
        </div>
        <div className={styles.buttonRight}>
          <Button text="Clear" color="--error-color" onClick={handleClearClick} />
          <Button text="Submit" onClick={handleSubmitClick} />
        </div>
      </div>
    </div>
  );
}

export default StoryWriting;
