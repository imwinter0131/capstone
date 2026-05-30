import React, { useEffect, useState } from "react";
import Login from "./Login";
import DatasetManage from "./DatasetManage";
import Projects from "./Projects";
import TrainingManage from "./TrainingManage";
import ThemeToggle from "./ThemeToggle";

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

  let page;

  if (path === "/projects") {
    if (!user?.user_id) {
      page = <Login />;
    } else {
      page = <Projects user={user} />;
    }
  } else if (path.startsWith("/projects/")) {
    if (!user?.user_id) {
      page = <Login />;
    } else {
      const pathParts = path.split("/").filter(Boolean);
      const projectId = pathParts[1];

      if (pathParts[2] === "preprocessing") {
        page = <DatasetManage user={user} projectId={projectId} />;
      } else if (pathParts[2] === "training") {
        page = <TrainingManage user={user} projectId={projectId} />;
      } else {
        page = <DatasetManage user={user} projectId={projectId} />;
      }
    }
  } else {
    page = <Login />;
  }

  return (
    <>
      <ThemeToggle />
      {page}
    </>
  );
}

export default App;
