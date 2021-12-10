import React, { Component } from 'react'
import Drawing from './canvas'

import "./style.css"

export class Sketchpage extends Component {
    constructor(props) {
        super(props)
        this.state = {}
    }
    
    render() {
        return (
            <div>
                <div>
                    <Drawing/>
                </div>
            </div>
        )
    }
}