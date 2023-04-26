import React from 'react';
import { render, fireEvent } from '@testing-library/react';
import Button from './Button';

describe('Button', () => {
  it('renders correctly with default props', () => {
    const { getByText } = render(<Button text="Click me!" />);
    expect(getByText('Click me!')).toBeInTheDocument();
  });

  it('calls onClick prop when clicked', () => {
    const handleClick = jest.fn();
    const { getByText } = render(<Button text="Click me!" onClick={handleClick} />);
    fireEvent.click(getByText('Click me!'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('applies custom class name', () => {
    const { getByText } = render(<Button text="Click me!" className="custom-button" />);
    expect(getByText('Click me!')).toHaveClass('custom-button');
  });
});
