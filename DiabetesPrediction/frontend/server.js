const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const app = express();

app.use(cors());
app.use(bodyParser.json());
app.use(express.static("public")); // Serve static files (HTML, CSS, JS)

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Frontend server running on port ${PORT}`);
});