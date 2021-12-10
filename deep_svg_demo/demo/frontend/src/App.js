import './App.css';
import {BrowserRouter, Route, Switch} from "react-router-dom";
import { Sketchpage } from "./sketchpage";

function App() {
  return (
    <div className="App">
    <BrowserRouter basename={process.env.PUBLIC_URL} onUpdate={() => window.scrollTo(0, 0)}>
      <Switch>
        <Route exact path='/' component={Sketchpage}/>
      </Switch>
    </BrowserRouter>
  </div>
  );
}

export default App;
