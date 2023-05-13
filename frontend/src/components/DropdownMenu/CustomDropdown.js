import React, { useState } from 'react';
import './CustomDropdown.css';

const CustomDropdown = ({ options, value, onChange }) => {
  const [isOpen, setIsOpen] = useState(false);

  const handleToggle = () => {
    setIsOpen(!isOpen);
  };

  const handleOptionClick = (option) => {
    onChange(option);
    setIsOpen(false);
  };

  return (
    <div className="dropdown">
      <button className="dropdown-toggle" onClick={handleToggle}>
        {value.label}
        {value.icon && <img src={value.icon} alt={value.label} className="dropdown-icon"/>}
      </button>
      {isOpen && (
        <div className="dropdown-menu">
          {options.map((option, index) => (
            <div
              key={index}
              className="dropdown-item"
              onClick={() => handleOptionClick(option)}
            >
              {option.label}
              <img src={option.icon} alt={option.label} className="dropdown-icon" />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default CustomDropdown;
