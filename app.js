import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js';

import {
  GLTFLoader
} from 'https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/loaders/GLTFLoader.js';


/* =========================================================
   3D VIEWER SETUP
========================================================= */

const host =
  document.getElementById('heart-container');

if (!host) {
  throw new Error(
    'heart-container element was not found in index.html'
  );
}


const scene =
  new THREE.Scene();

scene.background =
  new THREE.Color('#030b14');

scene.fog =
  new THREE.Fog(
    '#030b14',
    8,
    18
  );


const camera =
  new THREE.PerspectiveCamera(
    34,
    1,
    0.1,
    100
  );

camera.position.set(
  0,
  0.2,
  8.7
);


const renderer =
  new THREE.WebGLRenderer({
    antialias: true,
    alpha: false
  });

renderer.setPixelRatio(
  Math.min(
    window.devicePixelRatio || 1,
    1.8
  )
);

renderer.outputColorSpace =
  THREE.SRGBColorSpace;

renderer.toneMapping =
  THREE.ACESFilmicToneMapping;

renderer.toneMappingExposure =
  1.25;

renderer.shadowMap.enabled =
  true;

renderer.shadowMap.type =
  THREE.PCFSoftShadowMap;

host.appendChild(
  renderer.domElement
);


/* =========================================================
   ROOT
========================================================= */

const root =
  new THREE.Group();

scene.add(root);


/* =========================================================
   PREMIUM LIGHTING
========================================================= */

scene.add(
  new THREE.HemisphereLight(
    0xb6d6ff,
    0x240d18,
    2.0
  )
);


const keyLight =
  new THREE.DirectionalLight(
    0xffddd7,
    3.4
  );

keyLight.position.set(
  3,
  5,
  5
);

keyLight.castShadow = true;

scene.add(
  keyLight
);


const blueLight =
  new THREE.PointLight(
    0x287cff,
    22,
    12
  );

blueLight.position.set(
  -4,
  1,
  3
);

scene.add(
  blueLight
);


const pinkLight =
  new THREE.PointLight(
    0xff2458,
    13,
    10
  );

pinkLight.position.set(
  3,
  -2,
  2
);

scene.add(
  pinkLight
);


const fillLight =
  new THREE.DirectionalLight(
    0xff6b64,
    1.15
  );

fillLight.position.set(
  0,
  -4,
  2
);

scene.add(
  fillLight
);


/* =========================================================
   MODEL GROUPS
========================================================= */

const groups = {
  'Coronary Arteries': [],
  'Coronary Veins': [],
  'Chambers': [],
  'Aorta & Great Vessels': [],
  'Pericardium': [],
  'Nerves': []
};


/* =========================================================
   HEART MODEL
========================================================= */

let heartModel = null;
let heartMeshes = [];
let isHeartLoaded = false;
let baseHeartScale = 1.0;
let heartRotation = true;
let riskHighlight = false;


/* =========================================================
   LOADING ELEMENT
========================================================= */

const loadingEl =
  document.getElementById(
    'heart-loading'
  );


/* =========================================================
   GLB LOADER
========================================================= */

const gltfLoader =
  new GLTFLoader();


gltfLoader.load(

  '/anatomy/heart.glb',

  function (gltf) {

    heartModel =
      gltf.scene;


    /* -----------------------------------------
       BOUNDS
    ----------------------------------------- */

    const box =
      new THREE.Box3()
        .setFromObject(
          heartModel
        );

    const center =
      box.getCenter(
        new THREE.Vector3()
      );

    const size =
      box.getSize(
        new THREE.Vector3()
      );


    /* -----------------------------------------
       CENTER
    ----------------------------------------- */

    heartModel.position.sub(
      center
    );


    /* -----------------------------------------
       NORMALIZE SIZE
    ----------------------------------------- */

    const maxSize =
      Math.max(
        size.x,
        size.y,
        size.z
      );

    const scale =
      3.15 / maxSize;

    heartModel.scale.setScalar(
      scale
    );

    baseHeartScale =
      scale;


    /* -----------------------------------------
       MODEL MATERIALS
    ----------------------------------------- */

    heartModel.traverse(
      object => {

        if (!object.isMesh) {
          return;
        }

        object.castShadow =
          true;

        object.receiveShadow =
          true;

        heartMeshes.push(
          object
        );


        /*
          The selected GLB is a combined
          anatomical mesh, so the whole
          model is kept as one group.
        */

        groups[
          'Chambers'
        ].push(
          object
        );


        if (object.material) {

          object.material.roughness =
            0.35;

          object.material.metalness =
            0.015;


          if (
            'envMapIntensity'
            in object.material
          ) {

            object.material.envMapIntensity =
              1.15;

          }

        }

      }
    );


    root.add(
      heartModel
    );


    /* -----------------------------------------
       INITIAL ORIENTATION
    ----------------------------------------- */

    heartModel.rotation.x =
      THREE.MathUtils.degToRad(
        -4
      );

    heartModel.rotation.y =
      THREE.MathUtils.degToRad(
        8
      );

    heartModel.rotation.z =
      THREE.MathUtils.degToRad(
        -2
      );


    root.rotation.y =
      -0.10;


    isHeartLoaded =
      true;


    if (loadingEl) {
      loadingEl.style.display =
        'none';
    }


    console.log(
      'CardioVision 3D realistic heart loaded.'
    );

  },

  undefined,

  function (error) {

    console.error(
      'Heart GLB loading error:',
      error
    );


    if (loadingEl) {

      loadingEl.innerHTML = `
        <div style="
          display:flex;
          flex-direction:column;
          align-items:center;
          justify-content:center;
          gap:8px;
          text-align:center;
          padding:25px;
        ">

          <span style="
            color:#ff8295;
            font-size:13px;
            font-weight:600;
          ">
            Heart model could not be loaded.
          </span>

          <span style="
            color:#8ca2bf;
            font-size:11px;
          ">
            Check anatomy/heart.glb
          </span>

        </div>
      `;

    }

  }

);


/* =========================================================
   ILLUSTRATIVE CORONARY VESSEL OVERLAY
========================================================= */

function createVesselMaterial(
  color,
  emissive
) {

  return new THREE.MeshStandardMaterial({

    color,

    roughness: 0.26,

    metalness: 0.02,

    emissive,

    emissiveIntensity: 0.35

  });

}


const vesselMaterial =
  createVesselMaterial(
    0xff3948,
    0x350006
  );


function addTube(
  name,
  points,
  radius,
  material,
  groupName
) {

  const curve =
    new THREE.CatmullRomCurve3(
      points.map(
        p =>
          new THREE.Vector3(
            ...p
          )
      )
    );


  const geometry =
    new THREE.TubeGeometry(
      curve,
      Math.max(
        40,
        points.length * 10
      ),
      radius,
      10,
      false
    );


  const mesh =
    new THREE.Mesh(
      geometry,
      material
    );


  mesh.name =
    name;

  mesh.castShadow =
    true;

  mesh.receiveShadow =
    true;


  root.add(
    mesh
  );


  groups[
    groupName
  ].push(
    mesh
  );


  return mesh;

}


/* =========================================================
   LAD
========================================================= */

addTube(
  'LAD',

  [
    [0.05, 0.45, 0.51],
    [0.18, 0.12, 0.64],
    [0.24, -0.3, 0.67],
    [0.17, -0.72, 0.62],
    [0.28, -1.15, 0.43]
  ],

  0.032,

  vesselMaterial,

  'Coronary Arteries'
);


/* =========================================================
   LCX
========================================================= */

addTube(
  'LCX',

  [
    [0.36, 0.45, 0.48],
    [0.68, 0.31, 0.42],
    [0.85, 0.02, 0.38],
    [0.82, -0.32, 0.38]
  ],

  0.030,

  vesselMaterial,

  'Coronary Arteries'
);


/* =========================================================
   RCA
========================================================= */

addTube(
  'RCA',

  [
    [-0.52, 0.4, 0.57],
    [-0.73, 0.15, 0.59],
    [-0.65, -0.16, 0.61],
    [-0.4, -0.39, 0.59],
    [-0.08, -0.55, 0.54]
  ],

  0.032,

  vesselMaterial,

  'Coronary Arteries'
);


/* =========================================================
   LEFT CORONARY
========================================================= */

addTube(
  'Left Coronary Artery',

  [
    [0.13, 0.53, 0.47],
    [0.43, 0.44, 0.49],
    [0.69, 0.3, 0.43]
  ],

  0.028,

  vesselMaterial,

  'Coronary Arteries'
);


/* =========================================================
   RIGHT MARGINAL
========================================================= */

addTube(
  'Right marginal artery',

  [
    [-0.65, -0.12, 0.58],
    [-0.77, -0.48, 0.48],
    [-0.61, -0.82, 0.4]
  ],

  0.018,

  vesselMaterial,

  'Coronary Arteries'
);


/* =========================================================
   MOUSE CONTROLS
========================================================= */

let isDragging =
  false;

let previous = {
  x: 0,
  y: 0
};

let pointerButton =
  0;


renderer.domElement.addEventListener(
  'pointerdown',
  e => {

    isDragging =
      true;

    previous = {
      x: e.clientX,
      y: e.clientY
    };

    pointerButton =
      e.button;

    renderer.domElement
      .setPointerCapture(
        e.pointerId
      );

  }
);


renderer.domElement.addEventListener(
  'pointermove',
  e => {

    if (!isDragging) {
      return;
    }


    const dx =
      e.clientX -
      previous.x;

    const dy =
      e.clientY -
      previous.y;


    previous = {
      x: e.clientX,
      y: e.clientY
    };


    if (
      pointerButton === 2
    ) {

      root.position.x +=
        dx * 0.004;

      root.position.y -=
        dy * 0.004;

    } else {

      root.rotation.y +=
        dx * 0.008;

      root.rotation.x +=
        dy * 0.006;

    }

  }
);


renderer.domElement.addEventListener(
  'pointerup',
  () => {

    isDragging =
      false;

  }
);


renderer.domElement.addEventListener(
  'pointercancel',
  () => {

    isDragging =
      false;

  }
);


renderer.domElement.addEventListener(
  'contextmenu',
  e => {

    e.preventDefault();

  }
);


/* =========================================================
   ZOOM
========================================================= */

renderer.domElement.addEventListener(
  'wheel',

  e => {

    e.preventDefault();

    camera.position.z =
      Math.max(
        4.5,
        Math.min(
          12,
          camera.position.z +
          e.deltaY * 0.006
        )
      );

  },

  {
    passive: false
  }
);


/* =========================================================
   RESIZE
========================================================= */

function resize() {

  const width =
    host.clientWidth ||
    600;

  const height =
    host.clientHeight ||
    600;


  renderer.setSize(
    width,
    height,
    false
  );


  camera.aspect =
    width / height;


  camera.updateProjectionMatrix();

}


new ResizeObserver(
  resize
).observe(host);

resize();


/* =========================================================
   ANIMATION
========================================================= */

const clock =
  new THREE.Clock();


function animate() {

  requestAnimationFrame(
    animate
  );


  const time =
    clock.getElapsedTime();


  /* -----------------------------------------
     SUBTLE HEARTBEAT
  ----------------------------------------- */

  if (
    isHeartLoaded &&
    heartModel
  ) {

    const beat =
      1 +
      Math.sin(
        time *
        Math.PI *
        2.0
      ) *
      0.010;


    heartModel.scale.setScalar(
      baseHeartScale *
      beat
    );

  }


  /* -----------------------------------------
     SLOW AUTO ROTATION
  ----------------------------------------- */

  if (
    !isDragging &&
    heartRotation
  ) {

    root.rotation.y +=
      0.0008;

  }


  renderer.render(
    scene,
    camera
  );

}


animate();


/* =========================================================
   LABELS
========================================================= */

const labelNames = [
  'label-svc',
  'label-aorta',
  'label-pa',
  'label-rca',
  'label-lca',
  'label-lad',
  'label-lcx',
  'label-ivc'
];


document
  .getElementById(
    'labelsToggle'
  )
  ?.addEventListener(
    'click',

    e => {

      document.body
        .classList.toggle(
          'labels-off'
        );

      e.currentTarget
        .classList.toggle(
          'selected'
        );

    }
  );


/* =========================================================
   3D / 2D
========================================================= */

document
  .getElementById(
    'mode3d'
  )
  ?.addEventListener(
    'click',

    () => {

      document.body
        .classList.remove(
          'viewer-2d'
        );

      document
        .getElementById(
          'mode3d'
        )
        ?.classList.add(
          'selected'
        );

      document
        .getElementById(
          'mode2d'
        )
        ?.classList.remove(
          'selected'
        );

    }
  );


document
  .getElementById(
    'mode2d'
  )
  ?.addEventListener(
    'click',

    () => {

      document.body
        .classList.add(
          'viewer-2d'
        );

      document
        .getElementById(
          'mode2d'
        )
        ?.classList.add(
          'selected'
        );

      document
        .getElementById(
          'mode3d'
        )
        ?.classList.remove(
          'selected'
        );


      toast(
        '2D-style view is a visual preview, not a diagnostic image.'
      );

    }
  );


/* =========================================================
   RESET VIEW
========================================================= */

document
  .getElementById(
    'resetView'
  )
  ?.addEventListener(
    'click',
    resetView
  );


function resetView() {

  root.rotation.set(
    0,
    -0.10,
    0
  );


  root.position.set(
    0,
    0,
    0
  );


  camera.position.set(
    0,
    0.2,
    8.7
  );


  if (heartModel) {

    heartModel.rotation.set(

      THREE.MathUtils.degToRad(
        -4
      ),

      THREE.MathUtils.degToRad(
        8
      ),

      THREE.MathUtils.degToRad(
        -2
      )

    );

  }

}


/* =========================================================
   FULLSCREEN
========================================================= */

document
  .getElementById(
    'fullscreenBtn'
  )
  ?.addEventListener(
    'click',

    () => {

      const panel =
        document.querySelector(
          '.viewer-panel'
        );


      if (
        !document.fullscreenElement
      ) {

        panel
          ?.requestFullscreen?.();

      } else {

        document
          .exitFullscreen?.();

      }

    }
  );


/* =========================================================
   ROTATION BUTTON
========================================================= */

const rotationButton =
  document.getElementById(
    'toggle-rotation'
  );


if (rotationButton) {

  rotationButton
    .addEventListener(
      'click',

      () => {

        heartRotation =
          !heartRotation;


        rotationButton
          .textContent =
            heartRotation
              ? '⏸ Pause Rotation'
              : '▶ Resume Rotation';


        rotationButton
          .setAttribute(
            'aria-pressed',
            String(
              heartRotation
            )
          );

      }
    );

}


/* =========================================================
   LAYERS BUTTON
========================================================= */

document
  .getElementById(
    'layersBtn'
  )
  ?.addEventListener(
    'click',

    () => {

      const arteries =
        groups[
          'Coronary Arteries'
        ];


      const visible =
        arteries.some(
          mesh =>
            mesh.visible
        );


      arteries.forEach(
        mesh => {

          mesh.visible =
            !visible;

        }
      );


      toast(
        !visible
          ? 'Coronary vessel overlay shown'
          : 'Coronary vessel overlay hidden'
      );

    }
  );


/* =========================================================
   RISK HIGHLIGHT
========================================================= */

document
  .getElementById(
    'riskColorBtn'
  )
  ?.addEventListener(
    'click',

    () => {

      riskHighlight =
        !riskHighlight;


      if (riskHighlight) {

        vesselMaterial.color.setHex(
          0xff1744
        );


        vesselMaterial.emissive.setHex(
          0x57000a
        );


        vesselMaterial.emissiveIntensity =
          0.7;


        heartMeshes.forEach(
          mesh => {

            if (
              mesh.material &&
              mesh.material.emissive
            ) {

              mesh.material
                .emissive
                .setHex(
                  0x170003
                );


              if (
                mesh.material
                  .emissiveIntensity
                !== undefined
              ) {

                mesh.material
                  .emissiveIntensity =
                  0.18;

              }

            }

          }
        );


        toast(
          'Illustrative vessel risk highlight enabled'
        );

      } else {

        vesselMaterial.color.setHex(
          0xff3948
        );


        vesselMaterial.emissive.setHex(
          0x350006
        );


        vesselMaterial.emissiveIntensity =
          0.35;


        heartMeshes.forEach(
          mesh => {

            if (
              mesh.material &&
              mesh.material.emissive
            ) {

              mesh.material
                .emissive
                .setHex(
                  0x000000
                );


              if (
                mesh.material
                  .emissiveIntensity
                !== undefined
              ) {

                mesh.material
                  .emissiveIntensity =
                  0;

              }

            }

          }
        );


        toast(
          'Default anatomy appearance restored'
        );

      }

    }
  );


/* =========================================================
   STRUCTURE ROWS
========================================================= */

document
  .querySelectorAll(
    '.structure-row'
  )
  .forEach(
    row => {

      row.addEventListener(
        'click',

        () => {

          const groupName =
            row.dataset.structure;


          const meshes =
            groups[
              groupName
            ] || [];


          if (
            meshes.length === 0
          ) {

            toast(
              `${groupName} is part of the combined heart model.`
            );

            return;

          }


          const visible =
            meshes.some(
              mesh =>
                mesh.visible
            );


          meshes.forEach(
            mesh => {

              mesh.visible =
                !visible;

            }
          );


          row.classList.toggle(
            'muted-row',
            visible
          );


          const eye =
            row.querySelector(
              '.eye'
            );


          if (eye) {

            eye.textContent =
              visible
                ? '○'
                : '◉';

          }

        }
      );

    }
  );


/* =========================================================
   TABS
========================================================= */

document
  .querySelectorAll(
    '.tab'
  )
  .forEach(
    tab => {

      tab.addEventListener(
        'click',

        () => {

          document
            .querySelectorAll(
              '.tab'
            )
            .forEach(
              t =>
                t.classList.remove(
                  'selected'
                )
            );


          tab.classList.add(
            'selected'
          );


          const details =
            document.getElementById(
              'tabDetails'
            );


          const list =
            document.getElementById(
              'structureList'
            );


          if (
            tab.dataset.tab ===
            'structures'
          ) {

            details?.classList.add(
              'hidden'
            );

            list?.classList.remove(
              'hidden'
            );

          } else {

            list?.classList.add(
              'hidden'
            );

            details?.classList.remove(
              'hidden'
            );


            if (!details) {
              return;
            }


            if (
              tab.dataset.tab ===
              'layers'
            ) {

              details.innerHTML = `

                <b>
                  Visualization layers
                </b>

                <p>
                  Toggle the illustrative coronary vessel overlay
                  using the Layers control.
                </p>

                <p>
                  The downloaded heart is a combined anatomical GLB,
                  while LAD, LCX and RCA lines are prototype overlays.
                </p>

                <p>
                  Vessel colors are illustrative and are not derived
                  from a patient scan.
                </p>

              `;

            } else {

              details.innerHTML = `

                <b>
                  About this model
                </b>

                <p>
                  CardioVision 3D uses a realistic anatomical heart
                  GLB for the visual prototype.
                </p>

                <p>
                  The model is not a patient-specific scan and is not
                  a clinically validated anatomical representation.
                </p>

                <p>
                  Target vessels:
                  LAD · LCX · RCA
                </p>

              `;

            }

          }

        }
      );

    }
  );


/* =========================================================
   STRUCTURE SEARCH
========================================================= */

document
  .getElementById(
    'structureSearch'
  )
  ?.addEventListener(

    'input',

    e => {

      const query =
        e.target.value
          .toLowerCase()
          .trim();


      document
        .querySelectorAll(
          '.structure-row'
        )
        .forEach(
          row => {

            const name =
              row.dataset.structure
                ?.toLowerCase() ||
              '';


            row.style.display =
              name.includes(
                query
              )
                ? 'flex'
                : 'none';

          }
        );


      labelNames.forEach(
        className => {

          const element =
            document.querySelector(
              '.' +
              className
            );


          if (!element) {
            return;
          }


          element.style.opacity =
            !query ||
            element.textContent
              .toLowerCase()
              .includes(
                query
              )
              ? '1'
              : '.15';

        }
      );

    }
  );


/* =========================================================
   PREDICTION DIALOG
========================================================= */

const dialog =
  document.getElementById(
    'predictionDialog'
  );


document
  .getElementById(
    'tryPrediction'
  )
  ?.addEventListener(
    'click',

    () => {

      document
        .getElementById(
          'predictionResults'
        )
        ?.classList.add(
          'hidden'
        );


      if (
        dialog &&
        !dialog.open
      ) {

        dialog.showModal();

      }

    }
  );


/* =========================================================
   PREDICTION API
========================================================= */

document
  .getElementById(
    'predictionForm'
  )
  ?.addEventListener(

    'submit',

    async e => {

      e.preventDefault();


      const data =
        Object.fromEntries(
          new FormData(
            e.currentTarget
          ).entries()
        );


      const results =
        document.getElementById(
          'predictionResults'
        );


      if (!results) {
        return;
      }


      results.classList.remove(
        'hidden'
      );


      results.innerHTML = `
        <p>
          Connecting to prediction backend…
        </p>
      `;


      try {

        const response =
          await fetch(

            '/api/predict',

            {

              method: 'POST',

              headers: {
                'Content-Type':
                  'application/json'
              },

              body:
                JSON.stringify(
                  data
                )

            }

          );


        if (
          !response.ok
        ) {

          throw new Error(
            `Prediction API returned ${response.status}`
          );

        }


        const output =
          await response.json();


        renderResults(
          output,
          results
        );


      } catch (error) {

        console.error(
          'Prediction error:',
          error
        );


        results.innerHTML = `

          <h3>
            Backend not connected yet
          </h3>

          <p style="
            color:#9db0d0;
            line-height:1.6
          ">
            The interface is ready, but the
            prediction backend could not be reached.
            No medical risk has been estimated.
          </p>

          <div class="form-note">
            Start the Flask server and try again.
          </div>

        `;

      }

    }
  );


/* =========================================================
   RENDER PREDICTION
========================================================= */

function renderResults(
  output,
  element
) {

  if (
    output.demo ||
    !output.predictions
  ) {

    element.innerHTML = `

      <h3>
        ${
          output.message ||
          'Model not trained'
        }
      </h3>

      <p style="
        color:#9db0d0
      ">
        No clinical prediction was produced.
      </p>

    `;

    return;
  }


  window.lastPrediction =
    output;


  const rows =
    Object.entries(
      output.predictions
    );


  element.innerHTML = `

    <h3>

      Model output

      <span style="
        font-size:11px;
        color:#8ca2c9;
        font-weight:400
      ">

        · ${
          output.model ||
          'CardioVision3D trained pipeline'
        }

      </span>

    </h3>


    <p style="
      font-size:11px;
      color:#9bb0d1
    ">
      Educational output only — not a diagnosis.
    </p>


    <div class="result-grid">

      ${
        rows
          .map(
            ([key, value]) => {

              const percentage =
                Math.round(
                  Number(value) *
                  100
                );


              return `

                <div class="result-tile">

                  <span>
                    ${key}
                  </span>


                  <strong>
                    ${percentage}%
                  </strong>


                  <div
                    class="result-bar"
                  >

                    <i style="
                      width:${Math.max(
                        0,
                        Math.min(
                          100,
                          percentage
                        )
                      )}%
                    "></i>

                  </div>

                </div>

              `;

            }
          )
          .join('')
      }

    </div>


    ${
      output.report_id
        ? `

          <div style="
            margin-top:18px;
            padding-top:16px;
            border-top:
              1px solid
              rgba(
                126,
                157,
                197,
                .15
              );
          ">

            <div style="
              font-size:11px;
              color:#91a5bf;
              margin-bottom:10px;
            ">
              Patient report generated successfully
            </div>


            <div style="
              display:flex;
              flex-wrap:wrap;
              gap:10px;
            ">

              <a
                href="/api/reports/${encodeURIComponent(
                  output.report_id
                )}/csv"
                download
                class="report-download-btn"
              >
                Download CSV
              </a>


              <a
                href="/api/reports/${encodeURIComponent(
                  output.report_id
                )}/pdf"
                download
                class="report-download-btn"
              >
                Download PDF
              </a>

            </div>

          </div>

        `
        : ''
    }

  `;
}


/* =========================================================
   THEME
========================================================= */

document
  .getElementById(
    'themeBtn'
  )
  ?.addEventListener(
    'click',

    () =>
      document.body.classList.toggle(
        'light-mode'
      )

  );


/* =========================================================
   RIGHT PANEL
========================================================= */

document
  .getElementById(
    'hideRight'
  )
  ?.addEventListener(
    'click',

    () => {

      document
        .querySelector(
          '.rightbar'
        )
        ?.classList.add(
          'hidden'
        );


      document
        .getElementById(
          'showRight'
        )
        ?.classList.remove(
          'hidden'
        );

    }
  );


document
  .getElementById(
    'showRight'
  )
  ?.addEventListener(
    'click',

    () => {

      document
        .querySelector(
          '.rightbar'
        )
        ?.classList.remove(
          'hidden'
        );


      document
        .getElementById(
          'showRight'
        )
        ?.classList.add(
          'hidden'
        );

    }
  );


/* =========================================================
   NAVIGATION
========================================================= */

const navItems =
  [
    ...document.querySelectorAll(
      '.nav-item'
    )
  ];


const workspace =
  document.querySelector(
    '.workspace'
  );


const viewerPanel =
  document.querySelector(
    '.viewer-panel'
  );


const rightbar =
  document.querySelector(
    '.rightbar'
  );


let generatedPage =
  document.getElementById(
    'generatedPage'
  );


if (
  !generatedPage &&
  workspace
) {

  generatedPage =
    document.createElement(
      'section'
    );


  generatedPage.id =
    'generatedPage';


  generatedPage.className =
    'generated-page hidden';


  generatedPage.style.cssText =
    `
      grid-column:1/-1;
      min-width:0;
      padding:26px;
      border:
        1px solid
        rgba(
          126,
          157,
          197,
          .15
        );
      border-radius:18px;
      background:
        rgba(
          10,
          23,
          39,
          .92
        );
      color:#eaf2ff;
      min-height:360px;
    `;


  workspace.appendChild(
    generatedPage
  );

}


/* =========================================================
   PAGE CONTENT
========================================================= */

const pageContent = {

  reports: `

    <div class="generated-eyebrow">
      CARDIOVISION 3D / ANALYTICS
    </div>


    <h2>
      Reports & Model Evaluation
    </h2>


    <p class="generated-muted">
      Patient-specific prediction results and downloadable reports
      generated by the CardioVision3D trained pipeline.
    </p>


    <div id="reportsDynamicContent">

      <div class="generated-note">
        Loading prediction reports...
      </div>

    </div>

  `,


  settings: `

    <div class="generated-eyebrow">
      CARDIOVISION 3D / PREFERENCES
    </div>


    <h2>
      Settings
    </h2>


    <p class="generated-muted">
      Adjust the appearance and workspace options available in this prototype.
    </p>


    <div class="settings-list">

      <div>

        <div>

          <strong>
            Appearance
          </strong>

          <p class="generated-muted">
            Switch between the current dark theme and light theme.
          </p>

        </div>


        <button
          type="button"
          id="generatedThemeBtn"
        >
          Toggle theme
        </button>

      </div>


      <div>

        <div>

          <strong>
            Anatomy panel
          </strong>

          <p class="generated-muted">
            Show or hide the anatomy controls panel.
          </p>

        </div>


        <button
          type="button"
          id="generatedRightBtn"
        >
          Toggle anatomy panel
        </button>

      </div>


      <div>

        <div>

          <strong>
            3D view
          </strong>

          <p class="generated-muted">
            Return to the interactive heart viewer.
          </p>

        </div>


        <button
          type="button"
          id="generatedViewerBtn"
        >
          Open 3D viewer
        </button>

      </div>

    </div>

  `
};


/* =========================================================
   REPORTS LOADER
========================================================= */

async function loadReportsPage() {

  const container =
    document.getElementById(
      'reportsDynamicContent'
    );


  if (!container) {
    return;
  }


  try {

    const response =
      await fetch(
        '/api/reports'
      );


    if (!response.ok) {

      throw new Error(
        'Reports API failed'
      );

    }


    const data =
      await response.json();


    const reports =
      Array.isArray(data)
        ? data
        : (
            data.reports ||
            []
          );


    reports.sort(
      (a, b) =>
        new Date(
          b.created_at || 0
        ) -
        new Date(
          a.created_at || 0
        )
    );


    const latest =
      reports[0] ||
      window.lastPrediction ||
      null;


    let latestPredictionHTML =
      '';


    if (
      latest &&
      latest.predictions
    ) {

      latestPredictionHTML = `

        <section
          class="report-section"
        >

          <div
            class="report-section-title"
          >
            Latest Prediction
          </div>


          <div
            class="report-prediction-grid"
          >

            ${
              Object.entries(
                latest.predictions
              )
              .map(
                ([key, value]) => {

                  const percentage =
                    Math.round(
                      Number(value) *
                      100
                    );


                  return `

                    <div
                      class="report-prediction-card"
                    >

                      <span>
                        ${key}
                      </span>


                      <strong>
                        ${percentage}%
                      </strong>


                      <div
                        class="report-mini-bar"
                      >

                        <i style="
                          width:${Math.max(
                            0,
                            Math.min(
                              100,
                              percentage
                            )
                          )}%
                        "></i>

                      </div>

                    </div>

                  `;

                }
              )
              .join('')
            }

          </div>


          ${
            latest.report_id
              ? `

                <div
                  class="report-actions"
                >

                  <a
                    href="/api/reports/${encodeURIComponent(
                      latest.report_id
                    )}/csv"
                    download
                    class="report-download-btn"
                  >
                    Download CSV
                  </a>


                  <a
                    href="/api/reports/${encodeURIComponent(
                      latest.report_id
                    )}/pdf"
                    download
                    class="report-download-btn"
                  >
                    Download PDF
                  </a>

                </div>

              `
              : ''
          }

        </section>

      `;

    } else {

      latestPredictionHTML = `

        <section
          class="report-section"
        >

          <div class="generated-note">

            No prediction report has been generated yet.

            Run a prediction from the Prediction panel first.

          </div>

        </section>

      `;

    }


    const reportHistoryHTML =
      reports.length
        ? `

          <section
            class="report-section"
          >

            <div
              class="report-section-title"
            >
              Generated Patient Reports
            </div>


            <div
              class="report-history"
            >

              ${
                reports
                  .map(
                    report => {

                      const created =
                        report.created_at
                          ? new Date(
                              report.created_at
                            ).toLocaleString()
                          : 'Unknown time';


                      const predictions =
                        report.predictions ||
                        {};


                      return `

                        <article
                          class="report-history-card"
                        >

                          <div
                            class="report-history-top"
                          >

                            <div>

                              <strong>
                                Patient Prediction Report
                              </strong>


                              <small>
                                ${created}
                              </small>

                            </div>


                            <span
                              class="report-id"
                            >
                              ${
                                report.report_id ||
                                'REPORT'
                              }
                            </span>

                          </div>


                          <div
                            class="report-history-predictions"
                          >

                            ${
                              Object.entries(
                                predictions
                              )
                              .map(
                                ([key, value]) => `

                                  <div>

                                    <span>
                                      ${key}
                                    </span>

                                    <strong>
                                      ${Math.round(
                                        Number(
                                          value
                                        ) *
                                        100
                                      )}%
                                    </strong>

                                  </div>

                                `
                              )
                              .join('')
                            }

                          </div>


                          <div
                            class="report-actions"
                          >

                            ${
                              report.report_id
                                ? `

                                  <a
                                    href="/api/reports/${encodeURIComponent(
                                      report.report_id
                                    )}/csv"
                                    download
                                    class="report-download-btn"
                                  >
                                    CSV
                                  </a>


                                  <a
                                    href="/api/reports/${encodeURIComponent(
                                      report.report_id
                                    )}/pdf"
                                    download
                                    class="report-download-btn"
                                  >
                                    PDF
                                  </a>

                                `
                                : ''
                            }

                          </div>

                        </article>

                      `;

                    }
                  )
                  .join('')
              }

            </div>

          </section>

        `
        : '';


    container.innerHTML =
      latestPredictionHTML +
      reportHistoryHTML;


  } catch (error) {

    console.error(
      'Reports loading error:',
      error
    );


    container.innerHTML = `

      <div class="generated-note">

        Could not load saved reports from the backend.

        Make sure the Flask server is running.

      </div>

    `;

  }

}


/* =========================================================
   GENERATED PAGE
========================================================= */

function showGeneratedPage(
  page
) {

  if (!generatedPage) {

    toast(
      'This page could not be opened because the workspace container is missing.'
    );

    return;

  }


  viewerPanel?.classList.add(
    'hidden'
  );

  rightbar?.classList.add(
    'hidden'
  );


  generatedPage.classList.remove(
    'hidden'
  );


  generatedPage.innerHTML =
    (
      pageContent[page] ||
      pageContent.reports
    )
    +
    `

      <style>

        .generated-page h2{
          font-size:25px;
          margin:8px 0 12px;
          color:#f1f6ff;
        }


        .generated-eyebrow{
          font-size:10px;
          letter-spacing:2px;
          font-weight:800;
          color:#75aaff;
        }


        .generated-muted{
          font-size:13px;
          line-height:1.7;
          color:#91a5bf;
          max-width:760px;
        }


        .generated-note{
          margin-top:20px;
          padding:15px;
          border-radius:12px;

          background:
            rgba(
              255,
              185,
              75,
              .07
            );

          border:
            1px solid
            rgba(
              255,
              185,
              75,
              .18
            );

          font-size:12px;
          line-height:1.7;
          color:#d4c29e;
        }


        .settings-list{
          margin-top:24px;
          display:grid;
          gap:12px;
          max-width:850px;
        }


        .settings-list > div{
          display:flex;
          align-items:center;
          justify-content:space-between;
          gap:16px;
          padding:16px;

          border:
            1px solid
            rgba(
              126,
              157,
              197,
              .15
            );

          border-radius:12px;
          background:#0b1b2d;
        }


        .settings-list strong{
          display:block;
          margin-bottom:5px;
        }


        .settings-list button{
          flex-shrink:0;

          border:
            1px solid
            #2a4c70;

          background:#142b45;
          color:#eaf2ff;

          border-radius:9px;

          padding:
            10px 13px;

          cursor:pointer;
        }


        .report-section{
          margin-top:22px;
        }


        .report-section-title{
          font-size:15px;
          font-weight:700;
          color:#eaf2ff;
          margin-bottom:12px;
        }


        .report-prediction-grid{
          display:grid;
          grid-template-columns:
            repeat(4,1fr);
          gap:12px;
        }


        .report-prediction-card{
          padding:16px;

          border:
            1px solid
            rgba(
              126,
              157,
              197,
              .16
            );

          border-radius:13px;
          background:#0b1b2d;
        }


        .report-prediction-card span{
          display:block;
          font-size:11px;
          color:#8ba2be;
          margin-bottom:8px;
        }


        .report-prediction-card strong{
          font-size:24px;
          color:#f1f6ff;
        }


        .report-mini-bar{
          height:5px;
          margin-top:12px;
          border-radius:10px;
          background:#1d304b;
          overflow:hidden;
        }


        .report-mini-bar i{
          display:block;
          height:100%;
          border-radius:10px;

          background:
            linear-gradient(
              90deg,
              #3b8cff,
              #c546ff
            );
        }


        .report-actions{
          display:flex;
          flex-wrap:wrap;
          gap:10px;
          margin-top:14px;
        }


        .report-download-btn{
          display:inline-flex;
          align-items:center;
          justify-content:center;
          text-decoration:none;

          border:
            1px solid
            #2c4d73;

          background:#122943;
          color:#eaf2ff;

          border-radius:9px;

          padding:
            9px 13px;

          font-size:12px;
          font-weight:600;

          cursor:pointer;
          transition:.2s ease;
        }


        .report-download-btn:hover{
          background:#193756;
          border-color:#4c7db3;
        }


        .report-history{
          display:grid;
          gap:12px;
        }


        .report-history-card{
          padding:17px;

          border:
            1px solid
            rgba(
              126,
              157,
              197,
              .16
            );

          border-radius:13px;
          background:#0b1b2d;
        }


        .report-history-top{
          display:flex;
          align-items:flex-start;
          justify-content:space-between;
          gap:15px;
        }


        .report-history-top strong{
          display:block;
          color:#e8f0ff;
          font-size:14px;
        }


        .report-history-top small{
          display:block;
          color:#8196b3;
          font-size:11px;
          margin-top:5px;
        }


        .report-id{
          font-size:10px;
          color:#7189a8;
          word-break:break-all;
        }


        .report-history-predictions{
          display:grid;
          grid-template-columns:
            repeat(4,1fr);
          gap:8px;
          margin-top:14px;
        }


        .report-history-predictions > div{
          padding:10px;
          border-radius:9px;
          background:#0e2238;
        }


        .report-history-predictions span{
          display:block;
          font-size:10px;
          color:#8298b5;
        }


        .report-history-predictions strong{
          display:block;
          margin-top:4px;
          color:#e8f0ff;
          font-size:16px;
        }


        @media(max-width:700px){

          .report-prediction-grid,
          .report-history-predictions{

            grid-template-columns:
              repeat(2,1fr);

          }

        }


        @media(max-width:600px){

          .settings-list > div{

            align-items:flex-start;

            flex-direction:column;

          }

        }

      </style>

    `;


  if (
    page === 'reports'
  ) {

    loadReportsPage();

  }


  document
    .getElementById(
      'generatedThemeBtn'
    )
    ?.addEventListener(
      'click',

      () =>
        document.body.classList.toggle(
          'light-mode'
        )

    );


  document
    .getElementById(
      'generatedRightBtn'
    )
    ?.addEventListener(
      'click',

      () =>
        rightbar?.classList.toggle(
          'hidden'
        )

    );


  document
    .getElementById(
      'generatedViewerBtn'
    )
    ?.addEventListener(
      'click',

      () =>
        activatePage(
          'viewer'
        )

    );

}


/* =========================================================
   PAGE SWITCHING
========================================================= */

function activatePage(
  page
) {

  navItems.forEach(
    item => {

      item.classList.toggle(
        'active',

        item.dataset.page ===
          page ||

        (
          page === 'viewer' &&
          item.dataset.page ===
            'dashboard'
        )

      );

    }
  );


  if (
    page === 'prediction'
  ) {

    generatedPage?.classList.add(
      'hidden'
    );

    viewerPanel?.classList.remove(
      'hidden'
    );


    if (
      dialog &&
      !dialog.open
    ) {

      dialog.showModal();

    }

    return;

  }


  if (
    page === 'anatomy'
  ) {

    generatedPage?.classList.add(
      'hidden'
    );

    viewerPanel?.classList.remove(
      'hidden'
    );

    rightbar?.classList.remove(
      'hidden'
    );


    rightbar?.scrollIntoView({
      behavior: 'smooth',
      block: 'nearest'
    });


    toast(
      'Anatomy controls are open. Select a structure to show or hide it.'
    );

    return;

  }


  if (
    page === 'reports' ||
    page === 'settings'
  ) {

    showGeneratedPage(
      page
    );

    return;

  }


  generatedPage?.classList.add(
    'hidden'
  );

  viewerPanel?.classList.remove(
    'hidden'
  );


  if (
    page === 'viewer' ||
    page === 'dashboard'
  ) {

    rightbar?.classList.remove(
      'hidden'
    );


    viewerPanel?.scrollIntoView({
      behavior: 'smooth',
      block: 'nearest'
    });

  }

}


/* =========================================================
   NAV BUTTONS
========================================================= */

navItems.forEach(
  item => {

    item.addEventListener(
      'click',

      () =>
        activatePage(
          item.dataset.page ||
          'viewer'
        )

    );

  }
);


/* =========================================================
   TOAST
========================================================= */

function toast(
  message
) {

  const element =
    document.getElementById(
      'toast'
    );


  if (!element) {

    console.info(
      message
    );

    return;

  }


  element.textContent =
    message;


  element.classList.add(
    'show'
  );


  clearTimeout(
    window.__toastTimer
  );


  window.__toastTimer =
    setTimeout(
      () => {

        element.classList.remove(
          'show'
        );

      },
      2600
    );

}