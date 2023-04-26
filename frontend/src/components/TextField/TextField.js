// src/components/TextField/TextField.js

import React from 'react';
import styles from './TextField.module.css';

function TextField({ placeholder, value, onChange, multiline = false, rows = 1 }) {
  const TagName = multiline ? 'textarea' : 'input';

  return (
    <TagName
      className={styles.textField}
      placeholder={placeholder}
      value={value}
      onChange={onChange}
      rows={multiline ? rows : undefined}
    />
  );
}

export default TextField;



// import React from 'react';
// import styles from './TextField.module.css';

// function TextField(props) {
//   const { placeholder, ...otherProps } = props;

//   return (
//     <input
//       className={styles.textField}
//       type="text"
//       placeholder={placeholder}
//       {...otherProps}
//     />
//   );
// }

// export default TextField;
