prompt_perspective = """
    You are given a set of images sampled from a video about a human manipulating an articulated object.
    Please determine whether this video is from a first-person perspective or a third-person perspective.
    
    A first-person perspective means the video is filmed from the operator's point of view, usually showing the operator's arms or hands extending from the bottom or sides of the frame, and the viewpoint is aligned with the operator's head direction.
    A third-person perspective means the video is filmed from an observer's point of view, usually showing the whole or most of the operator's body, and the viewpoint is not aligned with the operator's head direction.
    
    Here are some judgment principles:
        1. If only one hand appears, it must be first-person perspective.
        2. If a human face appears, it must be third-person perspective.
        3. In first-person perspective, the hand(s) usually occupy a large area of the image.
        
    If it is first-person perspective, output only the number 1.
    If it is third-person perspective, output only the number 3.
    Do not output any other text or explanation.
"""

prompt = """
    ### Step 2: Frame-by-Frame Contact Reasoning
    The image contains K frames (horizontally merged) separated by black bars. 
    - Top row: RGB frames.
    - Bottom row: Depth frames (color gradient: blue=near, red=far).
    
    For each frame, analyze the 'Left Hand' and 'Right Hand' (identified in Step 1) separately using the following Chain of Thought:

    **Analysis Process for each hand per frame:**
    A. **Visibility Check**: Is the hand visible in this frame? If not, skip to next.
        (A.1) Left or Right hand may be occluded by the articulated object. I need to identify partial, occluded hand around object, not missing them.
    B. **Object contact estimation**: Is the hand close enough to contact the articulated object (but not background) in the RGB frame? 
        If the hand is clearly distant from the object or merely contact, mark 'contact: false' and skip to next.
        (B.1) I need to carefully identify the hand appearance, fully utilizing both RGB and depth.
        (B.2) I need to carefully determine if the hand is contacting the articulated object in a solid state, or it's in mere contact (which I should output FALSE)
    C. **Depth Map Verification (Critical Phase)**: 
       - Look at the bottom row (Depth map) corresponding to the hand's position.
       - **Reasoning**: Does the hand's depth color seamlessly merge with the object's depth color at the interaction point?
       - **Constraint**: Is there a sharp edge or color contrast separating them? If YES -> FALSE CONTACT.
       - **Decision**: Only mark 'contact: true' if the hand and object depth values merge without discontinuity.
    D. **Finger Analysis**:
       - If 'contact: true', identify specific fingers (thumb, index, middle) involved.
       - If a finger is occluded or ambiguous, exclude it.

    ### Step 3: Consistency & Final Decision
    - Review your frame-by-frame findings. 
    - Ensure the contact status transitions logically (e.g., hand approaches -> touches -> leaves) and is fully supported by the visual evidence in each frame independently.
    - Combining the neighbouring frames, make sure judge merely contact frame as FALSE contact.
    
    ### Step 4: Final Output Generation
    Generate the JSON output based on the analysis above.
    
    IMPORTANT: 
    - If you are not 100% sure about contact, you MUST judge as `false`.
    - Do not simply output the same status for all frames; look for changes.

    Output format (JSON only):
    {
        "frames_cnt": <number_of_frames>,
        "appeared": ["left", "right"],  // list only hands that appeared
        "contacts": [
            {
                "frame": <frame_number>, 
                "r_contact": <bool>, 
                "l_contact": <bool>,
                "r_fingers": ["thumb", "index", "middle"], // list valid fingers only, empty if no contact
                "l_fingers": []
            }
            // ... repeat for all K frames
        ]
    }
"""



Third_Person_Perspective_Hand_Hint =  (
    "### Step 1: Perspective Analysis & Hand Mapping (Chain of Thought)\n"
    "This video is Third-person perspective. You must determine the exact camera angle to identify hands correctly:\n"
    "1. **Analyze Operator Orientation**: Look at the person's body/head in the RGB frames.\n"
    "2. **Determine View Type & Arm Connectivity**:\n"
    "   - **Frontal/Side-Front View**: Camera faces the person. -> **Logic**: Mirror effect (Left Hand is on Right, Right Hand is on Left).\n"
    "   - **Rear/Side-Rear View**: Camera looks at the person's back or side-back. Use **Arm Connectivity** to identify hands:\n"
    "     * **Right Side-Rear**: Camera observes from the operator's right-back. The **Right Arm** is visibly connected to the body on the right. -> **Logic**: The hand connected to this visible right arm is the **Right Hand**. The other hand is the Left Hand.\n"
    "     * **Left Side-Rear**: Camera observes from the operator's left-back. The **Left Arm** is visibly connected to the body on the left. -> **Logic**: The hand connected to this visible left arm is the **Left Hand**. The other hand is the Right Hand.\n"
    "3. **Apply Mapping**: Based on the arm connections, strictly assign 'Left Hand' and 'Right Hand' labels before proceeding."
)

First_Perspective_Hand_Hint =  (
    "### Step 1: Perspective Analysis & Hand Mapping (Reasoning Chain of Thought Example)\n"
    "This video is First-person perspective. The camera mimics the operator's eyes. I need to determine carefully about hand side.\n"
    "1. If there's two hands, then the hand on the left side of the image is the Left Hand, and the hand on the right side is the Right Hand."
    "If only one hand appears, I need to determine carefully.\n"
    "1. **Standard Mapping**: Usually, the Left Hand enters from the logical left, and the Right Hand from the logical right.\n"
    "2. But I also need to examine the thumb direction to determine. Thumb of left hand is pointing right, and thumb of right hand is pointing left.\n"
    "So I can check the thumb direction to determine the hand side, especially when only one hand appears.\n"
    # "2. **Thumb Direction Check** (image coordinate system):\n"
    # "   - Imagine a coordinate plane on the hand.\n"
    # "   - If thumb points to Quadrants II/III (Leftward) -> Likely **Right Hand**.\n"
    # "   - If thumb points to Quadrants I/IV (Rightward) -> Likely **Left Hand**.\n"
    "3. **Apply Mapping**: Use these cues to strictly confirm the identity of any visible hands before proceeding."
)