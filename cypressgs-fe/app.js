const express = require('express');
const app = express();
const port = 7001;

app.use(express.static('public'));

app.get('/', (req, res) => {
  res.sendFile('/index.html');
});

app.listen(port, () => {
  console.log(`Server running at http://127.0.0.1:${port}`);
});
