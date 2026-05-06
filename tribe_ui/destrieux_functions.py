from __future__ import annotations

from copy import deepcopy


DEFAULT_REGION_FUNCTION = {
    "systems": ["Association cortex"],
    "summary": "Higher-order cortical processing; interpret with neighboring regions and the stimulus context.",
    "functions": [
        "Integrates information with nearby cortical networks.",
        "May reflect distributed attention, perception, memory, or task-context demands.",
    ],
    "stimuli": ["Context-dependent multimodal input"],
    "notes": "Atlas parcels are anatomical, not one-to-one mental modules.",
}


DESTRIEUX_FUNCTIONS = {
    "Medial_wall": {
        "systems": ["Atlas boundary"],
        "summary": "Medial wall is an atlas boundary/non-cortical surface label rather than a functional cortical parcel.",
        "functions": [
            "Marks cortex-adjacent medial-wall territory that should not be interpreted as a specific mental function.",
            "Activity here is best treated as anatomical context or model spillover near neighboring medial parcels.",
            "Use adjacent cingulate, paracentral, precuneus, and occipital parcels for functional interpretation.",
        ],
        "stimuli": ["Not functionally specific"],
        "notes": "Included for completeness because the atlas exposes it as a label.",
    },
    "G_and_S_frontomargin": {
        "systems": ["Prefrontal control", "Social cognition"],
        "summary": "Frontomarginal/frontopolar cortex supports abstract goals, prospective thinking, and social or value-based evaluation.",
        "functions": [
            "Maintains long-range goals and alternative courses of action.",
            "Supports counterfactual reasoning, future planning, and relational abstraction.",
            "Contributes to social inference and value-guided decisions.",
        ],
        "stimuli": ["Narrative goals", "Social cues", "Novel decisions", "Future-oriented language"],
    },
    "G_and_S_occipital_inf": {
        "systems": ["Visual recognition"],
        "summary": "Inferior occipital cortex participates in early-to-intermediate visual feature and object-form analysis.",
        "functions": [
            "Processes edges, contours, shape fragments, and object parts.",
            "Feeds ventral visual stream object and face recognition areas.",
            "Responds strongly to visually detailed video frames.",
        ],
        "stimuli": ["Objects", "Faces", "Scenes", "High-contrast motion"],
    },
    "G_and_S_paracentral": {
        "systems": ["Sensorimotor", "Body representation"],
        "summary": "Paracentral cortex represents lower-limb motor control and somatosensation on the medial surface.",
        "functions": [
            "Supports foot, leg, and trunk movement representations.",
            "Processes lower-body touch and proprioceptive signals.",
            "Can activate during observed or imagined body movement.",
        ],
        "stimuli": ["Walking", "Body motion", "Touch", "Action observation"],
    },
    "G_and_S_subcentral": {
        "systems": ["Sensorimotor", "Orofacial control"],
        "summary": "Subcentral cortex links inferior precentral and postcentral regions for face, mouth, tongue, and speech-articulation representations.",
        "functions": [
            "Represents orofacial movement and somatosensation.",
            "Contributes to articulation, swallowing, and facial action processing.",
            "Can respond to seen speech, gestures, or mouth movements.",
        ],
        "stimuli": ["Speech articulation", "Mouth movement", "Faces", "Touch"],
    },
    "G_and_S_transv_frontopol": {
        "systems": ["Prefrontal control"],
        "summary": "Transverse frontopolar cortex supports high-level planning, exploration, and integration of internally maintained goals.",
        "functions": [
            "Coordinates abstract, multi-step decisions.",
            "Supports exploration, uncertainty monitoring, and prospective memory.",
            "Links current evidence with internally generated plans.",
        ],
        "stimuli": ["Planning", "Uncertainty", "Complex instructions", "Goal shifts"],
    },
    "G_and_S_cingul-Ant": {
        "systems": ["Salience", "Cognitive control", "Affect"],
        "summary": "Anterior cingulate cortex tracks salience, conflict, effort, pain, autonomic arousal, and action selection.",
        "functions": [
            "Monitors conflict, error likelihood, and need for control.",
            "Links affective salience with behavioral responses.",
            "Participates in pain, effort, and autonomic regulation.",
        ],
        "stimuli": ["Conflict", "Errors", "Pain cues", "Emotionally salient moments"],
    },
    "G_and_S_cingul-Mid-Ant": {
        "systems": ["Cognitive control", "Action selection"],
        "summary": "Anterior mid-cingulate cortex supports effortful control, response selection, and motivated action.",
        "functions": [
            "Tracks task difficulty, effort, and action costs.",
            "Supports response selection under conflict.",
            "Couples cognitive control with motor preparation.",
        ],
        "stimuli": ["Demanding choices", "Action conflict", "Performance feedback", "Urgency"],
    },
    "G_and_S_cingul-Mid-Post": {
        "systems": ["Motor control", "Attention"],
        "summary": "Posterior mid-cingulate cortex links attention, body state, and motor control.",
        "functions": [
            "Coordinates motor readiness and orienting.",
            "Contributes to pain and somatic salience processing.",
            "Supports action monitoring in dynamic scenes.",
        ],
        "stimuli": ["Movement", "Threat", "Somatic cues", "Attention shifts"],
    },
    "G_cingul-Post-dorsal": {
        "systems": ["Default mode", "Spatial memory"],
        "summary": "Dorsal posterior cingulate cortex participates in internally directed attention, scene context, and autobiographical memory.",
        "functions": [
            "Supports environmental context and spatial orientation.",
            "Links memory retrieval with ongoing perception.",
            "Interacts with attention networks during shifts between internal and external focus.",
        ],
        "stimuli": ["Scenes", "Navigation", "Memory cues", "Narrative context"],
    },
    "G_cingul-Post-ventral": {
        "systems": ["Default mode", "Memory"],
        "summary": "Ventral posterior cingulate cortex is a core default-mode hub for self-relevant thought and episodic memory.",
        "functions": [
            "Supports autobiographical memory and self-referential evaluation.",
            "Integrates narrative meaning with personal relevance.",
            "Couples with medial temporal memory systems.",
        ],
        "stimuli": ["Personal narratives", "Memory cues", "Social meaning", "Rest-like reflection"],
    },
    "G_cuneus": {
        "systems": ["Visual cortex"],
        "summary": "Cuneus is medial occipital visual cortex involved in basic visual processing and spatial visual attention.",
        "functions": [
            "Processes contrast, orientation, and visual field structure.",
            "Supports visually guided attention and scene layout.",
            "Often activates with bright, structured, or moving visual input.",
        ],
        "stimuli": ["Visual contrast", "Motion", "Scenes", "Spatial layouts"],
    },
    "G_front_inf-Opercular": {
        "systems": ["Language", "Cognitive control"],
        "summary": "Inferior frontal operculum supports speech production, phonology, syntax, and response inhibition, especially in the dominant hemisphere.",
        "functions": [
            "Contributes to articulation and phonological working memory.",
            "Supports syntactic processing and controlled language production.",
            "Participates in inhibitory control and action stopping.",
        ],
        "stimuli": ["Speech", "Words", "Syntax", "Stop signals"],
    },
    "G_front_inf-Orbital": {
        "systems": ["Language semantics", "Valuation"],
        "summary": "Orbital inferior frontal cortex links semantic retrieval with reward, emotion, and value evaluation.",
        "functions": [
            "Supports controlled semantic retrieval and selection.",
            "Evaluates affective and reward value of stimuli.",
            "Contributes to flexible interpretation of ambiguous meaning.",
        ],
        "stimuli": ["Meaningful words", "Rewards", "Emotional faces", "Ambiguity"],
    },
    "G_front_inf-Triangul": {
        "systems": ["Language", "Cognitive control"],
        "summary": "Triangular inferior frontal cortex supports semantic selection, syntax, and controlled retrieval, typically left-dominant for language.",
        "functions": [
            "Selects among competing word meanings.",
            "Supports sentence-level syntactic and semantic processing.",
            "Contributes to verbal working memory and cognitive control.",
        ],
        "stimuli": ["Sentences", "Semantic ambiguity", "Inner speech", "Instructions"],
    },
    "G_front_middle": {
        "systems": ["Executive control", "Working memory"],
        "summary": "Middle frontal gyrus is central to working memory, attention control, rule maintenance, and goal-directed behavior.",
        "functions": [
            "Maintains task rules and information in working memory.",
            "Controls attention and resolves competing choices.",
            "Supports planning and monitoring of complex behavior.",
        ],
        "stimuli": ["Instructions", "Task switches", "Problem solving", "Complex narratives"],
    },
    "G_front_sup": {
        "systems": ["Executive control", "Motor planning"],
        "summary": "Superior frontal cortex supports executive control, self-monitoring, and supplementary motor planning.",
        "functions": [
            "Maintains goals and action plans.",
            "Supports self-monitoring and internally guided behavior.",
            "Participates in voluntary movement preparation.",
        ],
        "stimuli": ["Planning", "Self-reference", "Action sequences", "Cognitive demand"],
    },
    "G_Ins_lg_and_S_cent_ins": {
        "systems": ["Interoception", "Salience", "Taste"],
        "summary": "Long insular gyrus and central insular sulcus integrate body-state, visceral, pain, taste, and salience signals.",
        "functions": [
            "Represents internal body state and visceral sensation.",
            "Contributes to pain, disgust, taste, and emotional awareness.",
            "Helps detect salient events that require attention.",
        ],
        "stimuli": ["Body-state cues", "Pain", "Taste", "Disgust", "Salient events"],
    },
    "G_insular_short": {
        "systems": ["Salience", "Interoception", "Emotion"],
        "summary": "Short insular gyri support awareness of bodily feelings, salience detection, emotion, taste, and speech-related control.",
        "functions": [
            "Maps interoceptive and affective states.",
            "Flags salient sensory or emotional changes.",
            "Contributes to articulation and vocal effort control.",
        ],
        "stimuli": ["Emotion", "Taste", "Speech effort", "Bodily arousal"],
    },
    "G_occipital_middle": {
        "systems": ["Visual perception"],
        "summary": "Middle occipital gyrus supports visual object, motion, and spatial feature processing.",
        "functions": [
            "Processes visual form, texture, and motion features.",
            "Supports object and scene perception.",
            "Feeds both dorsal spatial and ventral recognition streams.",
        ],
        "stimuli": ["Objects", "Motion", "Scenes", "Visual detail"],
    },
    "G_occipital_sup": {
        "systems": ["Visual attention", "Dorsal stream"],
        "summary": "Superior occipital gyrus contributes to visual attention, spatial analysis, and motion-sensitive dorsal-stream processing.",
        "functions": [
            "Processes upper-level visual field and spatial relationships.",
            "Supports visual search and attention orienting.",
            "Contributes to motion and action-relevant visual analysis.",
        ],
        "stimuli": ["Motion", "Spatial layouts", "Visual search", "Action scenes"],
    },
    "G_oc-temp_lat-fusifor": {
        "systems": ["Ventral visual recognition"],
        "summary": "Lateral occipitotemporal/fusiform cortex supports high-level recognition of faces, bodies, words, objects, and visual categories.",
        "functions": [
            "Recognizes complex object forms and category structure.",
            "Includes face-, body-, and word-sensitive ventral visual territories.",
            "Links visual appearance with learned identity and meaning.",
        ],
        "stimuli": ["Faces", "Bodies", "Words", "Objects", "Brands"],
    },
    "G_oc-temp_med-Lingual": {
        "systems": ["Visual scenes", "Reading"],
        "summary": "Lingual occipitotemporal cortex supports visual shape, scene, color, and word-form processing.",
        "functions": [
            "Processes complex visual patterns and scene elements.",
            "Contributes to reading-related visual word processing.",
            "Supports color and object-form analysis.",
        ],
        "stimuli": ["Text", "Scenes", "Color", "Detailed images"],
    },
    "G_oc-temp_med-Parahip": {
        "systems": ["Memory", "Scene perception"],
        "summary": "Parahippocampal cortex supports scene recognition, contextual associations, navigation, and episodic memory.",
        "functions": [
            "Represents places, layouts, and environmental context.",
            "Links perceptual input to memory associations.",
            "Supports navigation and event context.",
        ],
        "stimuli": ["Places", "Rooms", "Landscapes", "Memory cues"],
    },
    "G_orbital": {
        "systems": ["Reward", "Emotion", "Decision making"],
        "summary": "Orbital frontal cortex evaluates reward, punishment, emotion, smell, taste, and social value.",
        "functions": [
            "Computes subjective value and outcome expectations.",
            "Updates choices when reward contingencies change.",
            "Integrates affective, olfactory, gustatory, and social signals.",
        ],
        "stimuli": ["Rewards", "Food", "Faces", "Social feedback", "Odors"],
    },
    "G_pariet_inf-Angular": {
        "systems": ["Semantic cognition", "Default mode", "Attention"],
        "summary": "Angular gyrus supports semantic integration, number concepts, social cognition, memory retrieval, and attention reorienting.",
        "functions": [
            "Integrates word, visual, and contextual meaning.",
            "Supports episodic retrieval and perspective taking.",
            "Contributes to number processing and attention shifts.",
        ],
        "stimuli": ["Stories", "Concepts", "Social scenes", "Numbers", "Unexpected events"],
    },
    "G_pariet_inf-Supramar": {
        "systems": ["Phonology", "Praxis", "Attention"],
        "summary": "Supramarginal gyrus supports phonological processing, sensorimotor integration, tool/action knowledge, and empathy for touch or pain.",
        "functions": [
            "Maintains speech sounds in phonological working memory.",
            "Integrates body, touch, and action representations.",
            "Supports attention reorienting and praxis.",
        ],
        "stimuli": ["Speech sounds", "Gestures", "Tool use", "Touch", "Observed pain"],
    },
    "G_parietal_sup": {
        "systems": ["Dorsal attention", "Visuomotor control"],
        "summary": "Superior parietal cortex supports spatial attention, reaching, visuomotor transformations, and working memory for locations.",
        "functions": [
            "Maintains spatial maps for attention and action.",
            "Guides eye, hand, and body movements.",
            "Supports mental rotation and spatial working memory.",
        ],
        "stimuli": ["Movement trajectories", "Spatial tasks", "Reaching", "Visual search"],
    },
    "G_postcentral": {
        "systems": ["Somatosensory"],
        "summary": "Postcentral gyrus is primary somatosensory cortex for touch, proprioception, body position, and tactile detail.",
        "functions": [
            "Maps touch and body-surface sensation.",
            "Processes proprioceptive and tactile features.",
            "Can activate during observed touch or body-focused content.",
        ],
        "stimuli": ["Touch", "Body contact", "Texture", "Proprioception"],
    },
    "G_precentral": {
        "systems": ["Motor"],
        "summary": "Precentral gyrus is primary motor cortex for voluntary movement and motor representations.",
        "functions": [
            "Controls contralateral voluntary movement.",
            "Represents hand, face, mouth, arm, trunk, and leg actions.",
            "Can respond during action observation and motor imagery.",
        ],
        "stimuli": ["Actions", "Gestures", "Speech articulation", "Sports", "Dance"],
    },
    "G_precuneus": {
        "systems": ["Default mode", "Visuospatial imagery"],
        "summary": "Precuneus supports self-referential thought, episodic memory, mental imagery, and visuospatial perspective.",
        "functions": [
            "Builds internal scenes and perspectives.",
            "Supports autobiographical memory and self-related processing.",
            "Contributes to attention and visuospatial imagery.",
        ],
        "stimuli": ["Scenes", "Personal meaning", "Memory", "Perspective shifts"],
    },
    "G_rectus": {
        "systems": ["Orbitomedial prefrontal", "Valuation"],
        "summary": "Gyrus rectus participates in reward, emotion, olfaction, social evaluation, and value-guided behavior.",
        "functions": [
            "Evaluates affective and reward value.",
            "Links smell, taste, and visceral signals with decisions.",
            "Supports social and emotional judgment.",
        ],
        "stimuli": ["Food", "Odors", "Rewards", "Social cues", "Emotional content"],
    },
    "G_subcallosal": {
        "systems": ["Mood", "Autonomic regulation"],
        "summary": "Subcallosal cingulate/medial frontal cortex is involved in mood regulation, autonomic state, and affective valuation.",
        "functions": [
            "Regulates affective and autonomic responses.",
            "Interacts with limbic and default-mode networks.",
            "Contributes to sadness, reward, and visceral feeling states.",
        ],
        "stimuli": ["Mood cues", "Emotional narratives", "Reward loss", "Body-state cues"],
    },
    "G_temp_sup-G_T_transv": {
        "systems": ["Auditory cortex"],
        "summary": "Transverse temporal/Heschl cortex contains primary auditory cortex for sound frequency, timing, and intensity analysis.",
        "functions": [
            "Processes basic acoustic features.",
            "Supports early speech and music analysis.",
            "Responds strongly to sound onsets and structured audio.",
        ],
        "stimuli": ["Speech sounds", "Music", "Sound onsets", "Environmental audio"],
    },
    "G_temp_sup-Lateral": {
        "systems": ["Auditory language", "Social perception"],
        "summary": "Lateral superior temporal cortex supports speech comprehension, voice processing, biological motion, and social audiovisual integration.",
        "functions": [
            "Analyzes spoken language and vocal identity.",
            "Integrates faces, voices, and biological motion.",
            "Supports narrative comprehension and social cue interpretation.",
        ],
        "stimuli": ["Speech", "Voices", "Faces with speech", "Biological motion", "Stories"],
    },
    "G_temp_sup-Plan_polar": {
        "systems": ["Auditory association"],
        "summary": "Planum polare supports higher-order auditory analysis, voice/speech context, and anterior temporal integration.",
        "functions": [
            "Processes complex sounds and auditory object qualities.",
            "Contributes to voice and speech-context analysis.",
            "Links auditory input with semantic and social meaning.",
        ],
        "stimuli": ["Voices", "Music", "Complex sounds", "Speech context"],
    },
    "G_temp_sup-Plan_tempo": {
        "systems": ["Auditory language"],
        "summary": "Planum temporale supports speech-sound analysis, auditory scene segregation, phonology, and sensorimotor mapping.",
        "functions": [
            "Maps speech sounds to phonological categories.",
            "Separates auditory streams in complex sound scenes.",
            "Supports speech perception and auditory-motor integration.",
        ],
        "stimuli": ["Speech", "Phonemes", "Competing sounds", "Music"],
    },
    "G_temporal_inf": {
        "systems": ["Visual semantics"],
        "summary": "Inferior temporal cortex supports high-level object recognition, visual semantics, and category knowledge.",
        "functions": [
            "Recognizes objects and visually presented concepts.",
            "Links visual form with semantic memory.",
            "Participates in face, body, object, and word recognition networks.",
        ],
        "stimuli": ["Objects", "Faces", "Text", "Animals/tools", "Visual categories"],
    },
    "G_temporal_middle": {
        "systems": ["Semantic language", "Social perception"],
        "summary": "Middle temporal gyrus supports lexical semantics, narrative comprehension, action meaning, and social audiovisual interpretation.",
        "functions": [
            "Retrieves word and object meaning.",
            "Integrates sentence and story context.",
            "Contributes to biological motion and social cue interpretation.",
        ],
        "stimuli": ["Words", "Stories", "Actions", "Voices", "Social scenes"],
    },
    "Lat_Fis-ant-Horizont": {
        "systems": ["Language", "Attention"],
        "summary": "Anterior horizontal lateral fissure territory borders inferior frontal and insular language/control regions.",
        "functions": [
            "Supports speech, semantic selection, and control processes near inferior frontal cortex.",
            "May reflect insula-opercular salience and articulation demands.",
            "Often interpreted with neighboring inferior frontal parcels.",
        ],
        "stimuli": ["Speech", "Semantic choices", "Salient events", "Articulation"],
    },
    "Lat_Fis-ant-Vertical": {
        "systems": ["Language", "Social communication"],
        "summary": "Anterior vertical lateral fissure territory borders inferior frontal speech, semantic, and social-communication networks.",
        "functions": [
            "Contributes to language production and semantic control near Broca-region cortex.",
            "Supports controlled retrieval and communicative action.",
            "May reflect adjacent frontal operculum and insula activation.",
        ],
        "stimuli": ["Speech", "Conversation", "Gestures", "Controlled language"],
    },
    "Lat_Fis-post": {
        "systems": ["Auditory", "Multisensory social perception"],
        "summary": "Posterior lateral fissure territory borders superior temporal auditory, speech, and audiovisual integration cortex.",
        "functions": [
            "Supports speech and environmental sound analysis.",
            "Integrates auditory input with visual social cues.",
            "Contributes to attention to voices and biological motion.",
        ],
        "stimuli": ["Voices", "Speech", "Face-voice combinations", "Environmental sounds"],
    },
    "Pole_occipital": {
        "systems": ["Early visual cortex"],
        "summary": "Occipital pole contains early visual cortex for central visual field detail, contrast, and form.",
        "functions": [
            "Processes fine visual detail near the foveal representation.",
            "Responds to brightness, contrast, edges, and motion.",
            "Supports visual input feeding object and scene pathways.",
        ],
        "stimuli": ["Bright images", "Edges", "Motion", "Fine detail", "Text"],
    },
    "Pole_temporal": {
        "systems": ["Semantic memory", "Social cognition"],
        "summary": "Temporal pole integrates semantic, social, emotional, and person-knowledge information.",
        "functions": [
            "Links names, people, objects, and concepts with rich meaning.",
            "Supports social knowledge and theory-of-mind in narratives.",
            "Integrates emotion with semantic memory.",
        ],
        "stimuli": ["People", "Names", "Stories", "Emotional/social scenes"],
    },
    "S_calcarine": {
        "systems": ["Primary visual cortex"],
        "summary": "Calcarine sulcus contains primary visual cortex for retinotopic visual input.",
        "functions": [
            "Processes basic visual features such as contrast, orientation, and spatial frequency.",
            "Maps the visual field retinotopically.",
            "Drives downstream visual recognition and attention pathways.",
        ],
        "stimuli": ["Visual input", "Contrast", "Edges", "Motion", "Flicker"],
    },
    "S_central": {
        "systems": ["Sensorimotor boundary"],
        "summary": "Central sulcus separates primary motor and somatosensory cortices and reflects hand, face, limb, and body sensorimotor activity.",
        "functions": [
            "Marks adjacent voluntary movement and somatosensory representations.",
            "Supports sensorimotor integration for action.",
            "Can activate with observed movement or tactile content.",
        ],
        "stimuli": ["Movement", "Touch", "Gestures", "Speech articulation"],
    },
    "S_cingul-Marginalis": {
        "systems": ["Cingulo-opercular control", "Motor attention"],
        "summary": "Marginal cingulate sulcus borders medial frontal and paracentral systems for control, attention, and action monitoring.",
        "functions": [
            "Supports performance monitoring and sustained control.",
            "Links medial motor planning with attention to body/action.",
            "May reflect adjacent cingulate and paracentral activity.",
        ],
        "stimuli": ["Action monitoring", "Effort", "Body movement", "Attention demand"],
    },
    "S_circular_insula_ant": {
        "systems": ["Salience", "Interoception"],
        "summary": "Anterior circular insular sulcus borders anterior insula regions for salience, affect, taste, and body-state awareness.",
        "functions": [
            "Detects salient emotional or bodily events.",
            "Supports subjective feeling and visceral awareness.",
            "Contributes to speech effort and cognitive-control switching.",
        ],
        "stimuli": ["Emotion", "Taste", "Pain", "Surprise", "Speech effort"],
    },
    "S_circular_insula_inf": {
        "systems": ["Interoception", "Somatosensory integration"],
        "summary": "Inferior circular insular sulcus borders insula-opercular regions involved in visceral, somatic, gustatory, and salience processing.",
        "functions": [
            "Integrates body, taste, and visceral sensations.",
            "Contributes to salience detection and affective awareness.",
            "Interacts with nearby opercular somatosensory regions.",
        ],
        "stimuli": ["Taste", "Pain", "Body-state cues", "Salient sounds/images"],
    },
    "S_circular_insula_sup": {
        "systems": ["Salience", "Sensorimotor control"],
        "summary": "Superior circular insular sulcus borders dorsal anterior insula and frontal operculum for salience, control, and articulation.",
        "functions": [
            "Supports switching between large-scale control networks.",
            "Contributes to speech articulation and effort monitoring.",
            "Integrates salience with motor-control demands.",
        ],
        "stimuli": ["Speech", "Attention shifts", "Effort", "Salient events"],
    },
    "S_collat_transv_ant": {
        "systems": ["Memory", "Ventral visual"],
        "summary": "Anterior transverse collateral sulcus borders medial temporal areas for object-context memory and visual-semantic associations.",
        "functions": [
            "Links visual input with memory associations.",
            "Supports object-place and context representations.",
            "May reflect parahippocampal and fusiform activity.",
        ],
        "stimuli": ["Objects in places", "Scenes", "Memory cues", "Visual categories"],
    },
    "S_collat_transv_post": {
        "systems": ["Scene perception", "Ventral visual"],
        "summary": "Posterior transverse collateral sulcus borders lingual, fusiform, and parahippocampal regions for visual scenes and object form.",
        "functions": [
            "Supports scene layout and visual context processing.",
            "Contributes to ventral-stream object and word-form analysis.",
            "Links visual detail with place/context representations.",
        ],
        "stimuli": ["Scenes", "Text", "Objects", "Landmarks"],
    },
    "S_front_inf": {
        "systems": ["Language", "Cognitive control"],
        "summary": "Inferior frontal sulcus separates inferior and middle frontal systems for language control, working memory, and attention.",
        "functions": [
            "Supports controlled semantic and phonological retrieval.",
            "Contributes to executive attention and task-rule maintenance.",
            "Often reflects neighboring inferior/middle frontal activation.",
        ],
        "stimuli": ["Language", "Instructions", "Conflict", "Working memory"],
    },
    "S_front_middle": {
        "systems": ["Executive control", "Attention"],
        "summary": "Middle frontal sulcus tracks dorsolateral prefrontal systems for working memory, planning, and top-down attention.",
        "functions": [
            "Maintains task goals and decision rules.",
            "Controls attention and monitors uncertainty.",
            "Supports planning and cognitive flexibility.",
        ],
        "stimuli": ["Problem solving", "Task switches", "Uncertainty", "Complex scenes"],
    },
    "S_front_sup": {
        "systems": ["Executive control", "Motor planning"],
        "summary": "Superior frontal sulcus borders superior and middle frontal cortex for goal maintenance, attention, and action planning.",
        "functions": [
            "Supports top-down control and monitoring.",
            "Participates in voluntary action preparation.",
            "Links internal goals with attention allocation.",
        ],
        "stimuli": ["Planning", "Attention demand", "Action preparation", "Rules"],
    },
    "S_interm_prim-Jensen": {
        "systems": ["Multimodal parietal association"],
        "summary": "Intermediate primate sulcus of Jensen lies in inferior parietal association cortex for multimodal attention, semantics, and action knowledge.",
        "functions": [
            "Integrates visual, spatial, language, and action information.",
            "Supports tool/action knowledge and attention reorienting.",
            "Contributes to semantic and numerical processing.",
        ],
        "stimuli": ["Tools", "Gestures", "Spatial cues", "Conceptual content"],
    },
    "S_intrapariet_and_P_trans": {
        "systems": ["Dorsal attention", "Numerical cognition", "Visuomotor"],
        "summary": "Intraparietal and transverse parietal sulci support spatial attention, eye/hand coordination, magnitude, and numerical processing.",
        "functions": [
            "Guides visuospatial attention and reaching.",
            "Represents quantity, magnitude, and numerical relationships.",
            "Supports visual working memory and action-relevant spatial maps.",
        ],
        "stimuli": ["Spatial motion", "Numbers", "Reaching", "Visual search", "Quantity"],
    },
    "S_oc_middle_and_Lunatus": {
        "systems": ["Visual motion", "Dorsal visual"],
        "summary": "Middle occipital/lunate sulcal cortex supports visual feature, motion, and spatial-form analysis.",
        "functions": [
            "Processes motion and object boundaries.",
            "Supports dorsal-stream spatial visual analysis.",
            "Contributes to scene and action perception.",
        ],
        "stimuli": ["Motion", "Objects", "Scenes", "Visual search"],
    },
    "S_oc_sup_and_transversal": {
        "systems": ["Visual attention", "Dorsal visual"],
        "summary": "Superior/transverse occipital sulci support visual attention, spatial mapping, and motion-sensitive processing.",
        "functions": [
            "Analyzes spatial relations and visual field structure.",
            "Supports orienting to visual events.",
            "Feeds parietal attention and visuomotor networks.",
        ],
        "stimuli": ["Moving objects", "Spatial layouts", "Visual attention", "Scene changes"],
    },
    "S_occipital_ant": {
        "systems": ["Visual association"],
        "summary": "Anterior occipital sulcus borders lateral occipital visual association cortex for object, motion, and scene features.",
        "functions": [
            "Processes intermediate visual form and motion cues.",
            "Links occipital visual analysis with temporal/parietal streams.",
            "Supports recognition of dynamic visual content.",
        ],
        "stimuli": ["Objects", "Motion", "Scenes", "Body movement"],
    },
    "S_oc-temp_lat": {
        "systems": ["Ventral visual recognition"],
        "summary": "Lateral occipitotemporal sulcus participates in high-level visual recognition of objects, bodies, words, and actions.",
        "functions": [
            "Processes complex object and body form.",
            "Supports visual category recognition and action perception.",
            "Links visual input with semantic identity.",
        ],
        "stimuli": ["Bodies", "Faces", "Objects", "Words", "Actions"],
    },
    "S_oc-temp_med_and_Lingual": {
        "systems": ["Visual scenes", "Ventral visual"],
        "summary": "Medial occipitotemporal/lingual sulcus supports visual pattern, word-form, scene, and color processing.",
        "functions": [
            "Analyzes detailed visual forms and patterns.",
            "Contributes to reading and scene perception.",
            "Supports color and object-context processing.",
        ],
        "stimuli": ["Text", "Scenes", "Color", "Detailed objects"],
    },
    "S_orbital_lateral": {
        "systems": ["Orbitofrontal valuation"],
        "summary": "Lateral orbital sulcus supports reward/punishment evaluation, social feedback, and flexible decision making.",
        "functions": [
            "Evaluates negative outcomes and changing contingencies.",
            "Supports inhibition of previously rewarded responses.",
            "Integrates affective and social decision signals.",
        ],
        "stimuli": ["Feedback", "Punishment", "Social evaluation", "Choice changes"],
    },
    "S_orbital_med-olfact": {
        "systems": ["Olfaction", "Reward", "Emotion"],
        "summary": "Medial orbital/olfactory sulcus links smell, flavor, visceral affect, and reward valuation.",
        "functions": [
            "Processes olfactory and flavor-related value.",
            "Links visceral state with affective decisions.",
            "Contributes to reward and pleasantness judgments.",
        ],
        "stimuli": ["Odors", "Food", "Pleasant/unpleasant cues", "Reward"],
    },
    "S_orbital-H_Shaped": {
        "systems": ["Orbitofrontal valuation"],
        "summary": "H-shaped orbital sulcus marks orbitofrontal cortex involved in reward, emotion, sensory value, and choice updating.",
        "functions": [
            "Represents subjective value across sensory domains.",
            "Updates decisions after reward or punishment.",
            "Integrates social, emotional, taste, and smell signals.",
        ],
        "stimuli": ["Rewards", "Food", "Social feedback", "Emotional images", "Odors"],
    },
    "S_parieto_occipital": {
        "systems": ["Dorsal visual", "Spatial attention"],
        "summary": "Parieto-occipital sulcus links visual cortex with parietal networks for spatial attention, navigation, and scene layout.",
        "functions": [
            "Processes spatial layout and visual orientation.",
            "Supports navigation and attention to visual scenes.",
            "Connects visual input with action-relevant spatial maps.",
        ],
        "stimuli": ["Scenes", "Navigation", "Spatial layouts", "Motion"],
    },
    "S_pericallosal": {
        "systems": ["Medial frontal/default mode"],
        "summary": "Pericallosal sulcus borders medial cingulate and callosal regions involved in self, memory, affect, and control depending on position.",
        "functions": [
            "Reflects nearby cingulate default-mode, control, and affective networks.",
            "Supports internally directed thought and action monitoring.",
            "May relate to memory, emotion, or body-state context.",
        ],
        "stimuli": ["Self-relevant content", "Memory cues", "Emotion", "Control demand"],
    },
    "S_postcentral": {
        "systems": ["Somatosensory"],
        "summary": "Postcentral sulcus borders somatosensory and parietal cortex for touch, proprioception, and body-aware spatial processing.",
        "functions": [
            "Processes tactile and proprioceptive information.",
            "Links body sensation with spatial attention.",
            "Can respond to observed touch and body actions.",
        ],
        "stimuli": ["Touch", "Body motion", "Texture", "Observed contact"],
    },
    "S_precentral-inf-part": {
        "systems": ["Motor", "Speech articulation"],
        "summary": "Inferior precentral sulcus borders motor and premotor representations for face, mouth, hand, and speech actions.",
        "functions": [
            "Supports orofacial, hand, and speech-related motor planning.",
            "Contributes to action observation and imitation.",
            "Links motor control with frontal language systems.",
        ],
        "stimuli": ["Speech", "Mouth movement", "Gestures", "Hand actions"],
    },
    "S_precentral-sup-part": {
        "systems": ["Premotor", "Dorsal attention"],
        "summary": "Superior precentral sulcus borders dorsal premotor cortex for reaching, eye-hand coordination, and action planning.",
        "functions": [
            "Plans arm, hand, and visually guided actions.",
            "Supports motor preparation and action selection.",
            "Interacts with spatial attention networks.",
        ],
        "stimuli": ["Reaching", "Sports", "Gestures", "Action preparation"],
    },
    "S_suborbital": {
        "systems": ["Orbitofrontal", "Emotion/value"],
        "summary": "Suborbital sulcus borders ventral prefrontal areas involved in affective valuation, reward, and social decision making.",
        "functions": [
            "Integrates affective value and decision context.",
            "Supports reward expectation and behavioral flexibility.",
            "May reflect nearby olfactory/orbitofrontal processing.",
        ],
        "stimuli": ["Rewards", "Emotional cues", "Social feedback", "Food/odor cues"],
    },
    "S_subparietal": {
        "systems": ["Default mode", "Parietal association"],
        "summary": "Subparietal sulcus borders precuneus and posterior cingulate systems for memory, self-related thought, and spatial context.",
        "functions": [
            "Supports episodic memory and internally generated scenes.",
            "Links spatial context with self-referential processing.",
            "Contributes to perspective and narrative integration.",
        ],
        "stimuli": ["Scenes", "Memory", "Self-relevant narratives", "Perspective shifts"],
    },
    "S_temporal_inf": {
        "systems": ["Ventral visual semantics"],
        "summary": "Inferior temporal sulcus supports object recognition, visual category knowledge, and semantic associations.",
        "functions": [
            "Links visual form with object meaning.",
            "Supports recognition of faces, bodies, tools, and words through nearby ventral stream regions.",
            "Contributes to visual-semantic memory.",
        ],
        "stimuli": ["Objects", "Faces", "Tools", "Text", "Visual categories"],
    },
    "S_temporal_sup": {
        "systems": ["Auditory language", "Social perception"],
        "summary": "Superior temporal sulcus supports speech, voice, audiovisual integration, biological motion, and social perception.",
        "functions": [
            "Integrates faces, voices, gestures, and biological motion.",
            "Supports speech comprehension and communicative intent.",
            "Contributes to theory-of-mind and dynamic social perception.",
        ],
        "stimuli": ["Speech", "Voices", "Faces", "Gestures", "Biological motion"],
    },
    "S_temporal_transverse": {
        "systems": ["Auditory cortex"],
        "summary": "Transverse temporal sulcus borders primary and secondary auditory cortex for acoustic feature and speech-sound processing.",
        "functions": [
            "Processes frequency, timing, and sound intensity.",
            "Supports early speech and music perception.",
            "Responds to auditory onsets and structured sounds.",
        ],
        "stimuli": ["Speech sounds", "Music", "Sound onsets", "Environmental audio"],
    },
}


LEFT_RIGHT_NOTES = {
    "L": "Left-sided activation is often more language, speech-production, symbolic, and praxis weighted, though lateralization varies by person.",
    "R": "Right-sided activation is often more spatial attention, prosody, face/voice, emotion, and global-context weighted, though lateralization varies by person.",
}


def region_function(label: str, hemi: str | None = None) -> dict[str, object]:
    info = deepcopy(DESTRIEUX_FUNCTIONS.get(label, DEFAULT_REGION_FUNCTION))
    if hemi in LEFT_RIGHT_NOTES:
        info["lateralization"] = LEFT_RIGHT_NOTES[hemi]
    info["atlas_label"] = label
    return info
