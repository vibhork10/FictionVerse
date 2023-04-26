import React from 'react';
import { render, fireEvent } from '@testing-library/react';
import DropdownMenu from './DropdownMenu';

describe('DropdownMenu', () => {
  it('renders correctly with default props', () => {
    const options = [
      { value: 'option1', label: 'Option 1' },
      { value: 'option2', label: 'Option 2' },
      { value: 'option3', label: 'Option 3' }
    ];
    const { getByLabelText } = render(
      <DropdownMenu options={options} aria-label="dropdown menu" />
    );
    expect(getByLabelText('dropdown menu')).toBeInTheDocument();
  });

  it('calls onChange prop when option is selected', () => {
    const options = [
      { value: 'option1', label: 'Option 1' },
      { value: 'option2', label: 'Option 2' },
      { value: 'option3', label: 'Option 3' }
    ];
    const handleChange = jest.fn();
    const { getByLabelText } = render(
      <DropdownMenu options={options} aria-label="dropdown menu" onChange={handleChange} />
    );
    fireEvent.change(getByLabelText('dropdown menu'), { target: { value: 'option2' } });
    expect(handleChange).toHaveBeenCalledTimes(1);
    expect(handleChange).toHaveBeenCalledWith('option2');
  });
});
