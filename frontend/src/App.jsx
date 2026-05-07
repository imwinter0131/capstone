import React, { useEffect, useState } from "react";
import Login from "./Login";
import DatasetManage from "./DatasetManage";
import PreprocessingManage from "./PreprocessingManage";
import Projects from "./Projects";
import TrainingManage from "./TrainingManage";

function getStoredUser() {
  try {
    const raw = localStorage.getItem("dlops_user");
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function App() {
  const [path, setPath] = useState(window.location.pathname);
  const [user, setUser] = useState(getStoredUser());

  useEffect(() => {
    const syncRoute = () => {
      setPath(window.location.pathname);
      setUser(getStoredUser());
    };

    window.addEventListener("popstate", syncRoute);
    window.addEventListener("dlops:navigate", syncRoute);

    return () => {
      window.removeEventListener("popstate", syncRoute);
      window.removeEventListener("dlops:navigate", syncRoute);
    };
  }, []);

  if (path === "/projects") {
    if (!user?.user_id) {
      return <Login />;
    }

    return <Projects user={user} />;
  }

  if (path.startsWith("/projects/")) {
    if (!user?.user_id) {
      return <Login />;
    }

    const pathParts = path.split("/").filter(Boolean);
    const projectId = pathParts[1];

    if (pathParts[2] === "preprocessing") {
      return <PreprocessingManage user={user} projectId={projectId} />;
    }

    if (pathParts[2] === "training") {
      return <TrainingManage user={user} projectId={projectId} />;
    }

    return <DatasetManage user={user} projectId={projectId} />;
  }

  return <Login />;
}

export default App;
