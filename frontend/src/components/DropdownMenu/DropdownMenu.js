import React from 'react';
import styles from './DropdownMenu.module.css';

function DropdownMenu(props) {
  const { options, ...otherProps } = props;

  return (
    <div className={styles.dropdownMenu}>
      <select className={styles.dropdown} {...otherProps}>
        {options.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export default DropdownMenu;
