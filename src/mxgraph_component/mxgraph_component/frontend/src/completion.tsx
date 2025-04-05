

export interface CompletionResponse {
    response: string;
}

export interface ErrorResponse {
    error: string;
}


/**
 * Function to get completion from the server
 * @param text - The input text to send to the server
 * @param serverUrl - The URL of the server (e.g., 'http://localhost:8421/complete')
 * @returns A promise that resolves to the completion response string
 */
export async function getCompletion(text: string, serverUrl: string = 'http://localhost:8421/complete'): Promise<string> {
    try {
        // Construct the full URL with query parameters
        const url = new URL(serverUrl);
        // const url = new URL(window.origin + ':8421/complete');
        url.searchParams.append('text', text);

        console.log(window.origin);
        console.log(url.toString());


        // Make the GET request using Fetch API
        const response = await fetch(url.toString(), {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        // Check if the response status is OK (status code 200-299)
        if (!response.ok) {
            // Attempt to parse the error message from the response
            const errorData: ErrorResponse = await response.json();
            throw new Error(`Server Error: ${response.status} - ${errorData.error}`);
        }

        // Parse the response data
        const data: CompletionResponse = await response.json();

        if (data.response) {
            return data.response;
        } else {
            throw new Error(`Unexpected response structure: ${JSON.stringify(data)}`);
        }
    } catch (error: any) {
        // Handle errors (network issues, server errors, etc.)
        throw new Error(`Error: ${error.message}`);
    }
}
