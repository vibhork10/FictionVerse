import React from 'react';
import styles from './Button.module.css';

function Button(props) {
  const { text, ...otherProps } = props;

  return (
    <button className={styles.button} {...otherProps}>
      {text}
    </button>
  );
}

export default Button;
