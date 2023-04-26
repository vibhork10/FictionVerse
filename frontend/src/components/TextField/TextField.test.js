import React from 'react';
import { render, fireEvent } from '@testing-library/react';
import TextField from './TextField';

describe('TextField', () => {
  it('renders correctly with default props', () => {
    const { getByPlaceholderText } = render(<TextField placeholder="Enter text" />);
    expect(getByPlaceholderText('Enter text')).toBeInTheDocument();
  });

  it('calls onChange prop when text is entered', () => {
    const handleChange = jest.fn();
    const { getByPlaceholderText } = render(
      <TextField placeholder="Enter text" onChange={handleChange} />
    );
    fireEvent.change(getByPlaceholderText('Enter text'), { target: { value: 'hello' } });
    expect(handleChange).toHaveBeenCalledTimes(1);
    expect(handleChange).toHaveBeenCalledWith('hello');
  });
});
