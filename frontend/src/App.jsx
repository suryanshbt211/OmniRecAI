import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {

  const [userId, setUserId] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);

  const loadRecommendations = async () => {

    try {

      const response =
        await axios.get(
          `http://localhost:8000/recommend/${userId}`
        );

      setRecommendations(
        response.data.recommendations
      );

    } catch (error) {

      console.error(error);

      alert(
        "Failed to load recommendations"
      );
    }
  };

  const loadMetrics = async () => {

    try {

      const response =
        await axios.get(
          "http://localhost:8000/metrics"
        );

      setMetrics(
        response.data
      );

    } catch (error) {

      console.error(error);
    }
  };

  const loadHealth = async () => {

    try {

      const response =
        await axios.get(
          "http://localhost:8000/health"
        );

      setHealth(
        response.data
      );

    } catch (error) {

      console.error(error);
    }
  };

  return (

    <div
      style={{
        padding: "30px",
        fontFamily: "Arial"
      }}
    >

      <h1>
        OmniRecAI Dashboard
      </h1>

      <hr />

      <h2>
        Recommendations
      </h2>

      <input
        value={userId}
        onChange={(e) =>
          setUserId(e.target.value)
        }
        placeholder="Enter User ID"
      />

      <button
        onClick={loadRecommendations}
        style={{
          marginLeft: "10px"
        }}
      >
        Get Recommendations
      </button>

      <br />
      <br />

      <table border="1">

        <thead>

          <tr>
            <th>Rank</th>
            <th>Item ID</th>
            <th>Category</th>
            <th>Price</th>
            <th>Rating</th>
            <th>Score</th>
          </tr>

        </thead>

        <tbody>

          {recommendations.map(
            (item) => (

              <tr
                key={item.item_id}
              >
                <td>{item.rank}</td>
                <td>{item.item_id}</td>
                <td>{item.category}</td>
                <td>{item.price}</td>
                <td>{item.rating}</td>
                <td>
                  {item.score.toFixed(4)}
                </td>
              </tr>

            )
          )}

        </tbody>

      </table>

      <hr />

      <h2>
        Health Status
      </h2>

      <button
        onClick={loadHealth}
      >
        Check Health
      </button>

      {health && (

        <pre>
          {JSON.stringify(
            health,
            null,
            2
          )}
        </pre>

      )}

      <hr />

      <h2>
        Metrics
      </h2>

      <button
        onClick={loadMetrics}
      >
        Load Metrics
      </button>

      {metrics && (

        <pre>
          {JSON.stringify(
            metrics,
            null,
            2
          )}
        </pre>

      )}

    </div>
  );
}

export default App;