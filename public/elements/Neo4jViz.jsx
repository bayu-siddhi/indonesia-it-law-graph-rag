import React, { useEffect } from 'react';

export default function Neo4jViz() {
    return (
        <iframe
            src={props.src}
            width="100%"
            height="500px"
            style={{ border: 'none' }}
        />
    );
}