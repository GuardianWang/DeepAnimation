import React, { useState } from 'react'
import { useSvgDrawing } from 'react-hooks-svgdrawing'
import axios from 'axios'
import "./style.css"
import { Image } from 'react-bootstrap'


const Drawing = ({history}) => {
    const [renderRef, draw] = useSvgDrawing({
        penWidth: 12
    })
    const path = ''//'/Users/yy/courses/csci2470/DeepAnimation/icons/icon_gif/'
    const [strokeSize, setStrokeSize] = useState(12)
    const [gif, setGif] = useState('welcome.gif')
    function handleSliderChange(value) {
        setStrokeSize(value)
        draw.changePenWidth(value)
    }
    
    const submitHandler = ({history}) => {
        const svg_value = draw.getSvgXML()
        // console.log((svg_value))        
        // Generate SVG file
        const fileName = "test.svg"
        const saveFile = async (blob) => {
            const a = document.createElement('a');
            a.download = fileName;
            a.href = URL.createObjectURL(blob);
            a.addEventListener('click', (e) => {
              setTimeout(() => URL.revokeObjectURL(a.href), 1000);
            });
            a.click();
          };
        const blob = new Blob([svg_value], {type : 'application/json'});
        saveFile(blob);
        async function getResult() {
            const { data } = await axios.get(
                `http://127.0.0.1:8000/api/${fileName}`
            )
            setGif(data)
        }
        getResult()

    }
    return (
        <div className="flex-column">
            <div>
                <div className="canvas" style={{height: window.innerHeight}} ref={renderRef} />
                
                <div className="toolbar flex-column">

                    <div className="small-padding">
                        Stroke width: {strokeSize}
                        <input type="range" min={1} max={60} value={strokeSize} onChange={(e) => handleSliderChange(e.target.value)} className="slider" id="myRange"/>
                    </div>
                    <div className="flex-row">
                        <Image src={path + gif} alt='loading...' fluid />
                    </div>
                    <div className="flex-row">
                        <button class="button-28" onClick={() => draw.undo()}>Undo</button>
                        <button class="button-28" onClick={() => draw.clear()}>Clear</button>
                        <button class="button-28" onClick={submitHandler}>Animate</button>
                    </div>
                    
                </div>
            </div>
        </div> 
    )
}

export default Drawing