const express = require('express')
const app = express()
const multer = require('multer')
const path = require('path')


app.use(express.static('./views'))
app.set('view engine', 'ejs');

var storage = multer.diskStorage({
    destination: path.resolve(__dirname ),
    filename: function (req, file, cb) {
      cb(null, 'words.jpg')
    }
  })
   
var upload = multer({ storage })

app.get('/',(req,res)=>{
    res.render('index',{showResult: false})
})


app.post('/run_script', upload.single('words'),(req,res)=>{
    console.log(req.body)
    const sentence = req.body.sentence
    const font_size = req.body.font_size

    var spwan = require('child_process').spawn;
    var process = spwan('python',['./Scripts/main.py', sentence, font_size])

    process.stdout.on('data', (data)=> {
        console.log(data)
        res.render('index',{showResult: true})
    })
    process.stderr.on('data', (data) => {
        console.log(`stderr: ${data}`);
    });
})


app.listen('3000',()=>{
    console.log('Listening  on port 3000')
})