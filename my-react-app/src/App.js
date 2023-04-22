import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [serverResponse, setServerResponse] = useState('');

  const handleSubmit = async () => {
    try {
      const response = await axios.post('http://localhost:8000/generate_story', {
        genre: 'Test genre',
        user_input: 'Test user input',
      });
      setServerResponse(response.data.story);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <button onClick={handleSubmit}>Submit</button>
      <p>{serverResponse}</p>
    </div>
  );
}

export default App;
