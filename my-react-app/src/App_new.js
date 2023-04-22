import React, { useState } from 'react';
import './App.css';

function App() {
  const [userInput, setUserInput] = useState('');
  const [genre, setGenre] = useState('');
  const [serverResponse, setServerResponse] = useState('');

  const handleSubmit = async () => {
    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        genre: genre,
        user_input: userInput
      })
    };

    const response = await fetch('http://127.0.0.1:8000/generate_story', requestOptions);
    const data = await response.json();
    setServerResponse(data.story);
  };

  return (
    <div className="App">
      <h1>Story Generator</h1>
      <input
        type="text"
        value={genre}
        onChange={(e) => setGenre(e.target.value)}
        placeholder="Enter genre"
      />
      <br />
      <textarea
        value={userInput}
        onChange={(e) => setUserInput(e.target.value)}
        placeholder="Enter your story input"
        rows={5}
      />
      <br />
      <button onClick={handleSubmit}>Submit</button>
      <h2>Generated Story</h2>
      <p>{serverResponse}</p>
    </div>
  );
}

export default App;