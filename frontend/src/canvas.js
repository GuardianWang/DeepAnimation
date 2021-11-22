import React, { useState } from 'react'
import { useSvgDrawing } from 'react-hooks-svgdrawing'

import "./style.css"

const Drawing = () => {
    const [renderRef, draw] = useSvgDrawing({
        penWidth: 12
    })
    const [strokeSize, setStrokeSize] = useState(12)

    function handleSliderChange(value) {
        setStrokeSize(value)
        draw.changePenWidth(value)
    }

    return (
        <div className="flex-column">
            <div className="canvas" style={{height: window.innerHeight}} ref={renderRef} />
            <div className="toolbar flex-column">
                <div className="small-padding">
                    Stroke width: {strokeSize}
                    <input type="range" min={1} max={60} value={strokeSize} onChange={(e) => handleSliderChange(e.target.value)} className="slider" id="myRange"/>
                </div>
                <div className="flex-row">
                    <button class="button-28" onClick={() => draw.undo()}>Undo</button>
                    <button class="button-28" onClick={() => draw.clear()}>Clear</button>
                    <button class="button-28" onClick={() => alert(draw.getSvgXML())}>Animate</button>
                </div>
            </div>
        </div> 
    )
}

export default Drawing